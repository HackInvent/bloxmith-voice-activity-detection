"""Classify continuous Opus speech locally; downstream blocks own interruption policy."""
from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from .neural_vad import FRAME_SAMPLES, THRESHOLDS, SileroSpeech, verified_model

from bloxsmith_app.block_api import (
    BlockDefinition, BlockRuntimeContext, BlockRuntimeListenerContext, BlockRuntimeOutput,
    BlockRuntimePreparation, BlockRuntimePreparationContext, BlockRuntimeResult, RuntimeAudioFrame,
    render_inspector_template, render_node_card_template,
)

DEFAULTS = {"aggressiveness": 2, "speech_start_ms": 160, "silence_ms": 500,
            "min_level_dbfs": -48, "drain_timeout_sec": 5}
BOUNDS = {"aggressiveness": (0, 3), "speech_start_ms": (20, 500), "silence_ms": (100, 2000),
          "min_level_dbfs": (-80, -20), "drain_timeout_sec": (1, 20)}
RATE, FRAME_BYTES, MAX_BUFFER = 16000, FRAME_SAMPLES * 2, 8 * 1024 * 1024
_ABSENT = object()


def _config(raw: Mapping | None) -> dict:
    """Normalize finite integer settings; compiler metadata never becomes VAD configuration."""
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("Configuration VAD invalide.")
    if set(raw or {}) - set(DEFAULTS) - {"position", "runtime_path", "runtime_path_label"}:
        raise ValueError("Paramètre VAD inconnu.")
    result = {}
    for key, default in DEFAULTS.items():
        value = (raw or {}).get(key, default)
        try:
            number = float(value)
            integer = int(number)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{key} doit être un entier.") from None
        low, high = BOUNDS[key]
        if isinstance(value, bool) or integer != number or not low <= integer <= high:
            raise ValueError(f"{key} doit être compris entre {low} et {high}.")
        result[key] = integer
    return result


def _command(raw: Any) -> dict:
    """Accept bounded producer start/stop with exact final frame/byte counts and an abort flag."""
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > 4096:
            raise ValueError("Commande VAD trop volumineuse.")
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError):
            raise ValueError("command_in attend un JSON start/stop du producteur audio.") from None
    if not isinstance(raw, Mapping) or raw.get("action") not in {"start", "stop"}:
        raise ValueError("command_in attend start ou stop, pas une commande d’interruption.")
    allowed = {"action", "stream_id"} if raw["action"] == "start" else {"action", "stream_id", "frame_count", "byte_count", "aborted"}
    if set(raw) - allowed or not isinstance(raw.get("stream_id"), str) or not 1 <= len(raw["stream_id"]) <= 128:
        raise ValueError("Commande VAD invalide : stream_id et champs du cycle audio attendus.")
    result = dict(raw)
    if result["action"] == "stop":
        for field in ("frame_count", "byte_count"):
            if type(result.get(field)) is not int or not 0 <= result[field] <= 2**53 - 1:
                raise ValueError("Stop VAD exige frame_count et byte_count entiers positifs ou nuls.")
        if not isinstance(result.get("aborted", False), bool):
            raise ValueError("aborted doit être un booléen.")
        result.setdefault("aborted", False)
    return result


def _failure(message: str, stream_id: str = "") -> BlockRuntimeResult:
    """Expose bounded diagnostics without audio bytes or fabricated speech events."""
    return BlockRuntimeResult(status="failed", error=message, last_message=message,
        logs=[f"[voice-activity-detection] {message}"], metadata={"voice_activity_detection": {
            "state": "failed", "stream_id": stream_id}})


class _Detector:
    """Confirm neural speech scores and AC energy on the decoded sample clock, not volume alone."""

    def __init__(self, config: dict, stream_id: str, emit, diagnostic=None):
        """Own neural history and transitions; optional diagnostics carry numbers, never audio."""
        self.vad = SileroSpeech()
        self.config, self.stream_id, self.emit = config, stream_id, emit
        self.diagnostic = diagnostic
        self.last_diagnostic_at = time.monotonic()
        self.threshold = THRESHOLDS[config["aggressiveness"]]
        self.release_threshold = self.threshold - 0.15
        self.score = 0.0
        self.level_dbfs = -120.0
        self.pcm = bytearray()
        self.samples = self.voiced_samples = self.quiet_samples = self.start_sample = 0
        self.utterance_id = ""
        self.min_power = (32768 * 10 ** (config.get("min_level_dbfs", DEFAULTS["min_level_dbfs"]) / 20)) ** 2

    def _event(self, event: str, reason: str) -> None:
        """End offsets include classified silence, never wall-clock gaps or padding."""
        self.emit({"event": event, "reason": reason, "stream_id": self.stream_id,
            "utterance_id": self.utterance_id, "event_id": uuid4().hex,
            "audio_start_ms": self.start_sample * 1000 // RATE, "audio_end_ms": self.samples * 1000 // RATE,
            "detection": self.measurement()})

    def measurement(self) -> dict:
        """Describe the actual model decision; its score is not a calibrated certainty or speaker id."""
        return {"engine": "silero_v6.2.1_onnx", "speech_score": round(self.score, 4),
                "level_dbfs": round(self.level_dbfs, 1), "speech_threshold": self.threshold,
                "release_threshold": self.release_threshold,
                "confirmation_ms": self.config["speech_start_ms"],
                "audio_processed_ms": self.samples * 1000 // RATE}

    def _frame(self, data: bytes, samples: int) -> None:
        """Require sustained neural speech plus real AC energy, with distinct onset/release scores.

        Loud non-speech cannot pass on energy alone. Measuring each window also
        excludes quiet classifier tails and DC bias. Final padding never qualifies
        a new onset, contributes energy, or advances the source audio clock.
        """
        pcm = array("h", data[:samples * 2])
        if sys.byteorder != "little":
            pcm.byteswap()
        mean = sum(pcm) / samples if samples else 0
        power = max(0, sum(value * value for value in pcm) / samples - mean * mean) if samples else 0
        self.score = self.vad.probability(data)
        self.level_dbfs = max(-120.0, 10 * math.log10(power / 32768**2)) if power else -120.0
        threshold = self.release_threshold if self.utterance_id else self.threshold
        voiced = self.score >= threshold and power >= self.min_power
        if samples < FRAME_SAMPLES and not self.utterance_id:
            voiced = False
        self.samples += samples
        if not self.utterance_id:
            if voiced:
                if not self.voiced_samples:
                    self.start_sample = self.samples - samples
                self.voiced_samples += samples
                if self.voiced_samples * 1000 >= self.config["speech_start_ms"] * RATE:
                    self.utterance_id = uuid4().hex
                    self.quiet_samples = 0
                    self._event("speech_started", "voice")
            else:
                self.voiced_samples = 0
        else:
            self.quiet_samples = 0 if voiced else self.quiet_samples + samples
            if self.quiet_samples * 1000 >= self.config["silence_ms"] * RATE:
                self._event("speech_stopped", "silence")
                self.utterance_id = ""
                self.voiced_samples = self.quiet_samples = 0
        # Quiet input must be diagnosable without publishing graph events or flooding logs.
        now = time.monotonic()
        if self.diagnostic is not None and now - self.last_diagnostic_at >= 5:
            self.last_diagnostic_at = now
            self.diagnostic({"stream_id": self.stream_id, **self.measurement()})

    def feed(self, data: bytes) -> None:
        """Classify full PCM frames and retain at most one incomplete native frame."""
        self.pcm.extend(data)
        while len(self.pcm) >= FRAME_BYTES:
            frame = bytes(self.pcm[:FRAME_BYTES])
            del self.pcm[:FRAME_BYTES]
            self._frame(frame, FRAME_BYTES // 2)

    def finish(self) -> None:
        """Finalize only a verified producer stop; inactivity and abort do not enter here."""
        if len(self.pcm) % 2:
            raise ValueError("Dernier échantillon PCM incomplet.")
        if self.pcm:
            self._frame(bytes(self.pcm).ljust(FRAME_BYTES, b"\0"), len(self.pcm) // 2)
            self.pcm.clear()
        if self.utterance_id:
            self._event("speech_stopped", "stream_ended")
            self.utterance_id = ""


class _Capture:
    """Own one bounded nonblocking FFmpeg pipeline and optional producer count reconciliation."""

    def __init__(self, config: dict, stream_id: str, emit, diagnostic=None):
        """Allocate local state; decoding starts on the first complete container signature."""
        self.config, self.stream_id = config, stream_id
        self.detector = _Detector(config, stream_id, emit, diagnostic)
        self.pending = bytearray()
        self.process = None
        self.frames = self.bytes = 0
        self.sequence = self.profile = self.stop = self.stopped_at = self.closing_at = None

    def feed(self, frame: RuntimeAudioFrame) -> None:
        """Reject gaps, profile changes and queue limits before retaining encoded bytes."""
        profile = (frame.source_id, frame.codec, frame.sample_rate_hz, frame.channels)
        if frame.codec != "opus" or frame.sample_rate_hz != 48000 or frame.channels not in {1, 2}:
            raise ValueError("VAD attend Opus WebM/Ogg, horloge 48 kHz, mono ou stéréo.")
        if (self.sequence is not None and frame.sequence != self.sequence + 1) or (self.profile and self.profile != profile):
            raise ValueError("Trame VAD manquante ou profil audio modifié ; redémarrez la source.")
        if self.closing_at is not None or not frame.payload or len(frame.payload) > 524288:
            raise ValueError("Trame VAD vide, tardive ou trop volumineuse.")
        if len(self.pending) + len(frame.payload) > MAX_BUFFER:
            raise ValueError("Tampon VAD saturé : le décodage ne suit plus la source.")
        self.profile, self.sequence = profile, frame.sequence
        self.frames += 1
        self.bytes += len(frame.payload)
        self.pending.extend(frame.payload)
        self.complete()

    def finish(self, command: dict) -> None:
        """Record exact producer totals; network inactivity is never a speech ending."""
        if command["aborted"]:
            raise ValueError("Source audio interrompue ; aucun faux événement de fin de parole n’est émis.")
        if self.stop is not None and self.stop != command:
            raise ValueError("Commandes stop VAD contradictoires.")
        if self.stop is None:
            self.stop, self.stopped_at = command, time.monotonic()
        self.complete()

    def complete(self) -> bool:
        """Finish only after the independently routed audio/data agree on final counts."""
        if self.stop is None:
            return False
        if self.frames > self.stop["frame_count"] or self.bytes > self.stop["byte_count"]:
            raise ValueError("Totaux audio VAD incohérents après stop.")
        return self.frames == self.stop["frame_count"] and self.bytes == self.stop["byte_count"]

    def _open(self) -> None:
        """Restrict FFmpeg to Opus on stdin and PCM on stdout: no network or user files."""
        container = {b"OggS": "ogg", b"\x1aE\xdf\xa3": "matroska"}.get(bytes(self.pending[:4]))
        if not container:
            raise ValueError("En-tête Opus absent : commencez au début du conteneur WebM/Ogg.")
        self.process = subprocess.Popen([
            shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-probesize", "32", "-analyzeduration", "0", "-protocol_whitelist", "pipe",
            "-f", container, "-c:a", "opus", "-i", "pipe:0", "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(RATE), "-acodec", "pcm_s16le", "-f", "s16le", "-flush_packets", "1", "pipe:1",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        for handle in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(handle.fileno(), False)

    def pump(self) -> bool:
        """Advance finite IO batches, drain stderr, and report fully decoded explicit EOF."""
        now = time.monotonic()
        if self.stopped_at is not None and now - self.stopped_at > self.config["drain_timeout_sec"]:
            raise ValueError("Flux VAD incomplet après stop ou décodeur trop lent.")
        if self.process is None:
            if len(self.pending) >= 4:
                self._open()
            elif self.complete():
                if self.pending:
                    raise ValueError("En-tête audio VAD tronqué.")
                return True
            else:
                return False
        process = self.process
        with suppress(BlockingIOError):
            os.read(process.stderr.fileno(), 4096)
        if self.pending:
            with suppress(BlockingIOError):
                written = os.write(process.stdin.fileno(), self.pending[:65536])
                del self.pending[:written]
        if not self.pending and self.complete() and self.closing_at is None:
            process.stdin.close()
            self.closing_at = now
        try:
            pcm = os.read(process.stdout.fileno(), 6400)
        except BlockingIOError:
            return False
        if pcm:
            self.detector.feed(pcm)
            return False
        if process.poll() is None:
            return False
        if process.returncode != 0 or self.closing_at is None:
            raise ValueError("Le décodeur VAD s’est arrêté : audio invalide ou tronqué.")
        self.detector.finish()
        return True

    def close(self) -> None:
        """Reap this capture promptly; cancellation never fabricates a speech ending."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=.15)
            for handle in (self.process.stdin, self.process.stdout, self.process.stderr):
                handle.close()
        self.pending.clear()


# FB1 - Fixed Opus input, optional lifecycle input, JSON events and autonomous discovery.
# FB2 - Listen from Run with dependency checks; simulation performs no process/audio IO.
# FB3 - Neural speech classification, energy qualification, hysteresis and source-clock offsets.
# FB4 - Bounded continuous decoding, explicit stop totals and safe error/cancel cleanup.
# FB5 - Own validated accessible settings, diagnostics and end-user documentation.
class VoiceActivityDetectionBlock(BlockDefinition):
    """Detect local speech; a downstream Python block owns interruption policy."""
    kind = "voice_activity_detection"

    def _ports(self, context: Any) -> None:
        """Validate context port identities independently of their unchanged visual order.

        Missing, duplicate or altered ports still fail; both inputs stay optional.
        """
        for ports, expected in ((context.input_ports, ((1, "audio_in", "audio_stream"), (2, "command_in", "message"))),
                                (context.output_ports, ((1, "events_out", "message"),))):
            by_id = {port.id: port for port in ports}
            if (len(ports) != len(expected) or set(by_id) != {port_id for port_id, _, _ in expected}
                    or any((by_id[port_id].name, getattr(by_id[port_id], "transport", "message")) != (name, transport)
                           for port_id, name, transport in expected)):
                raise ValueError("VAD nécessite audio_in, command_in et events_out fixes.")
        if any(p.required or p.multiplicity != "one" or getattr(p, "execution_requirement", "not_required_for_execution") != "not_required_for_execution" for p in context.input_ports):
            raise ValueError("Les deux entrées VAD doivent rester facultatives à multiplicité un.")

    def prepare_runtime(self, context: BlockRuntimePreparationContext) -> BlockRuntimePreparation:
        """Validate without IO; active listening does not wait for Play or commands."""
        self._ports(context)
        _config(context.config)
        return BlockRuntimePreparation(listen_on_run=context.runtime_mode == "zeromq_active")

    def initialize_runtime(self, context: BlockRuntimeContext) -> BlockRuntimeResult:
        """Check prerequisites without installing anything or starting a decoder."""
        if context.runtime_mode == "zeromq_active":
            if not shutil.which("ffmpeg") or any(importlib.util.find_spec(name) is None for name in ("numpy", "onnxruntime")):
                return _failure("VAD requiert FFmpeg, NumPy et ONNX Runtime ; installez blocs/voice_activity_detection/requirements.txt avec l’interpréteur du serveur.")
            try:
                verified_model()
            except ValueError as exc:
                return _failure(str(exc))
        return BlockRuntimeResult(last_message="VAD Silero prêt : reconnaissance locale de parole au premier flux audio.")

    def execute_runtime(self, context: BlockRuntimeContext) -> BlockRuntimeResult:
        """Forward only fresh lifecycle data; simulation starts no detector or IO."""
        try:
            self._ports(context)
            _config(context.config)
            if context.runtime_mode != "zeromq_active":
                return BlockRuntimeResult(status="skipped", last_message="Simulation : aucune détection vocale ni émission JSON.")
            raw = _ABSENT
            if context.input_events:
                for event in context.input_events:
                    if event.input_port_id == 2 and event.input_port_name == "command_in":
                        raw = event.value
            else:
                attribute = context.input_attribute("command_in")
                if attribute is not None and attribute.status == "updated":
                    raw = attribute.value
            if raw is _ABSENT:
                return BlockRuntimeResult(status="skipped", last_message="VAD à l’écoute de audio_in.")
            sender = context.services.get("runtime_listener")
            if sender is None:
                raise ValueError("Listener VAD indisponible : Stop puis Run.")
            sender.send(_command(raw))
            return BlockRuntimeResult(last_message="Cycle audio transmis au VAD.")
        except (ValueError, TypeError) as exc:
            return _failure(str(exc))

    def listen_runtime(self, context: BlockRuntimeListenerContext) -> None:
        """Supervise independent captures; decoded silence, never network inactivity, ends speech."""
        config = _config(context.config)
        captures, retired = {}, deque(maxlen=128)

        def emit(event: dict) -> None:
            """Publish a typed speech event through the framework-owned result mailbox."""
            measure = event["detection"]
            context.emit_result(BlockRuntimeResult(outputs=[BlockRuntimeOutput(port_id=1, port_name="events_out",
                value=json.dumps(event), content_type="application/json")],
                last_message="Parole détectée." if event["event"] == "speech_started" else "Fin de parole détectée.",
                logs=[f"[vad-speech] {event['event']} audio={event['audio_end_ms']}ms "
                      f"score={measure['speech_score']} niveau={measure['level_dbfs']}dBFS "
                      f"seuil={measure['speech_threshold']} confirmation={measure['confirmation_ms']}ms"],
                metadata={self.kind: {"state": event["event"], **event}}))

        def diagnostic(measure: dict) -> None:
            """Refresh numeric diagnostics at most once every five seconds per capture, with no graph output."""
            context.emit_result(BlockRuntimeResult(metadata={self.kind: {"diagnostics": measure}}))

        def capture(stream_id: str) -> _Capture:
            """Admit at most four sessions, including stops awaiting their final frames."""
            if stream_id not in captures:
                if len(captures) >= 4:
                    raise ValueError("Maximum quatre captures VAD : reliez start/stop du producteur.")
                captures[stream_id] = _Capture(config, stream_id, emit, diagnostic)
            return captures[stream_id]

        def retire(stream_id: str) -> None:
            """Release sessions and ignore late frames carrying a retired stream identifier."""
            item = captures.pop(stream_id, None)
            if item:
                item.close()
            retired.append(stream_id)

        try:
            audio = context.services.get("runtime_audio_streams")
            if audio is None or not audio.available:
                raise ValueError("Reliez audio_in à Microphone Stream.audio_out ou une autre source Opus.")
            while not context.stop_requested():
                incoming = context.receive_command(timeout_sec=0)
                if incoming is not None:
                    command = _command(incoming.payload)
                    stream_id = command["stream_id"]
                    if stream_id not in retired:
                        try:
                            item = capture(stream_id)
                            if command["action"] == "stop":
                                item.finish(command)
                        except Exception as exc:
                            context.emit_result(_failure(str(exc), stream_id))
                            retire(stream_id)
                for _ in range(16):
                    frame = audio.receive_port("audio_in", timeout_sec=0)
                    if frame is None:
                        break
                    if frame.stream_id not in retired:
                        try:
                            if sum(len(item.pending) for item in captures.values()) + len(frame.payload) > MAX_BUFFER:
                                raise ValueError("Tampon global VAD saturé.")
                            capture(frame.stream_id).feed(frame)
                        except Exception as exc:
                            context.emit_result(_failure(str(exc), frame.stream_id))
                            retire(frame.stream_id)
                for stream_id, item in list(captures.items()):
                    try:
                        if item.pump():
                            retire(stream_id)
                    except Exception as exc:
                        context.emit_result(_failure(str(exc), stream_id))
                        retire(stream_id)
                time.sleep(.005)
        except Exception as exc:
            if not context.stop_requested():
                context.emit_result(_failure(str(exc)))
        finally:
            for item in captures.values():
                item.close()

    def ui_assets(self, surface: str = "modal") -> list[dict[str, str]]:
        """Declare owned settings assets; no browser audio participant belongs to this block."""
        if surface in {"modal", "inspector_panel"}:
            return [{"kind": "css", "path": "assets/css/block_ui.css"}, {"kind": "js", "path": "assets/js/common.js"},
                    {"kind": "js", "path": f"assets/js/{'block_modal' if surface == 'modal' else surface}.js"}]
        return []

    def _settings_html(self, node: dict) -> str:
        """Place speech timing first and explain all tuning with associated labels."""
        values = _config(node.get("config"))
        labels = {"speech_start_ms": "Parole minimale (ms)", "silence_ms": "Silence avant fin de parole (ms)",
                  "min_level_dbfs": "Seuil sonore minimal (dBFS)",
                  "aggressiveness": "Exigence de parole Silero (0–3)", "drain_timeout_sec": "Attente après stop (s)"}
        return ''.join(f'<label>{label}<input type="number" data-vad-setting="{key}" value="{values[key]}" '
            f'min="{BOUNDS[key][0]}" max="{BOUNDS[key][1]}" step="1" required /></label>' for key, label in labels.items())

    def render_node_card(self, *, node: dict, payload: dict | None = None) -> dict:
        """Describe speech activity without exposing audio or claiming speaker identification."""
        return render_node_card_template(block=self, node=node, node_classes=["voice-activity-node"],
                                        replacements={"title": str(node.get("title") or self.default_title())})

    def render_modal(self, *, node: dict, payload: dict | None = None) -> dict:
        """Render a native opaque panel with reachable actions and contained diagnostics."""
        template = (self.directory / "block_modal.html").read_text(encoding="utf-8")
        template = template.replace("{{ settings_html }}", self._settings_html(node))
        return {"html": self._render_generic_modal_template(template=template, node=node, payload=payload or {}),
                "context": {"node_id": str(node.get("id") or ""), "node_kind": self.kind}}

    def render_inspector_panel(self, *, node: dict, payload: dict | None = None) -> dict:
        """Expose durable settings and voice/echo caveats in the native inspector."""
        template = (self.directory / "inspector_panel.html").read_text(encoding="utf-8")
        return {"html": render_inspector_template(template=template, node={**node, "kind": self.kind, "type": self.kind},
            payload=payload, replacements={"settings_html": self._settings_html(node)}), "context": {"full_panel": True}}

    def handle_ui_action(self, *, node: dict, action: str, values: dict, payload: dict | None = None) -> dict:
        """Save one validated title/settings snapshot without modifying a running detector."""
        if action != "save_properties":
            return super().handle_ui_action(node=node, action=action, values=values, payload=payload)
        try:
            title = values.get("title", node.get("title") or self.default_title())
            if not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
                raise ValueError("Le nom doit contenir de 1 à 200 caractères.")
            settings = values.get("config", {})
            if not isinstance(settings, Mapping) or set(settings) - set(DEFAULTS):
                raise ValueError("Réglages VAD inconnus.")
            return {"node_patch": {"title": title.strip(), "config": _config({**(node.get("config") or {}), **settings})}}
        except (ValueError, TypeError) as exc:
            return {"error": str(exc)}
