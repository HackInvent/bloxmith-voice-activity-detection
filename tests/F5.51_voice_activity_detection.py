#!/usr/bin/env python3
"""FB1–FB5: actual local Silero speech detection, real graph modes and actual-shell properties."""
from __future__ import annotations

from array import array
from dataclasses import replace
from functools import lru_cache
import json
import math
from random import Random
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from blocs.voice_activity_detection.block import VoiceActivityDetectionBlock, DEFAULTS, _config, _command, _Detector, _Capture
from blocs.microphone_stream.block import MicrophoneStreamBlock
from blocs.registry import get_block_definition
from bloxsmith_app.block_api import BlockRuntimeContext, RuntimeAudioFrame
from bloxsmith_app.block_runtime import BlockInputEvent
from bloxsmith_app.graph import WorkflowGraph
from bloxsmith_app.orchestrator import WorkflowOrchestrator
from ui_smoke_common import create_project_api, graph_payload, project_editor_url, run_playwright_smoke
from block_test_artifacts import artifact_path
from block_test_packages import install_test_package, surface_payload
from playwright.sync_api import expect as expect_ui

BLOCK = VoiceActivityDetectionBlock()


def context(mode="zeromq_active", **values):
    """Build a public context using exactly the new block manifest ports."""
    return BlockRuntimeContext(run_id="vad-test", node_id="vad", kind=BLOCK.kind, title=BLOCK.default_title(),
        config=dict(DEFAULTS), runtime_mode=mode, root_dir=ROOT,
        input_ports=tuple(SimpleNamespace(**p) for p in BLOCK.default_inputs()),
        output_ports=tuple(SimpleNamespace(**p) for p in BLOCK.default_outputs()), **values)


def until(check, message, timeout=5):
    """Wait on an observable bounded condition, never indefinitely."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.005)
    raise AssertionError(message)


@lru_cache
def pcm_speech(text="Please stop the answer now"):
    """Synthesize original test speech offline with FFmpeg flite; no recording or API is used."""
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"flite=text='{text}':voice=slt", "-af", "adelay=300,apad=pad_dur=1",
        "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"],
        capture_output=True, check=True, timeout=10).stdout


@lru_cache
def encoded(container="webm", channels=1, *, noise=False):
    """Encode synthetic voice or clicks to the actual graph formats in mono/stereo."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1",
        "-i", "pipe:0", "-ar", "48000", "-ac", str(channels), "-c:a", "libopus", "-f", container]
    command += ["-page_duration", "100000"] if container == "ogg" else ["-cluster_time_limit", "100"]
    return subprocess.run([*command, "pipe:1"], input=noise_pcm(12000, clicks=True) if noise else pcm_speech(),
                          capture_output=True, check=True, timeout=10).stdout


def frame(data, sequence=1, stream_id="capture", channels=1):
    """Build immutable validated public audio metadata with no implicit command payload."""
    return RuntimeAudioFrame(message_id=f"frame-{sequence}", run_id="vad-test", topic="vad.test", source_id="micro",
        stream_id=stream_id, payload=data, codec="opus", sample_rate_hz=48000, channels=channels,
        sequence=sequence, timestamp_ms=1)


def test_contracts():
    """FB1/FB2/FB5: discovery, immutable preparation, safe simulation and explicit dependency failures."""
    assert isinstance(get_block_definition(BLOCK.kind), VoiceActivityDetectionBlock)
    assert BLOCK.model["runtime"]["active_execution_policy"] == "on_each_event"
    for mode in ("centralized", "zeromq_active"):
        ctx = context(mode)
        with patch("subprocess.Popen") as popen:
            assert BLOCK.prepare_runtime(ctx).listen_on_run == (mode == "zeromq_active")
            result = BLOCK.execute_runtime(ctx)
            assert result.status == "skipped" and not result.outputs
            popen.assert_not_called()
        invalid = replace(ctx, input_ports=ctx.input_ports[:1])
        assert BLOCK.execute_runtime(invalid).status == "failed"
    with patch("importlib.util.find_spec", return_value=None), patch("subprocess.Popen") as popen:
        assert BLOCK.initialize_runtime(context()).status == "failed"
        assert BLOCK.initialize_runtime(context("centralized")).status == "success"
        popen.assert_not_called()
    for setting in ({"aggressiveness": True}, {"silence_ms": "nan"}, {"speech_start_ms": 0},
                    {"drain_timeout_sec": 99}, {"unknown": 1}, {"speech_start_ms": 20.1},
                    {"min_level_dbfs": -81}, {"min_level_dbfs": -19}, {"min_level_dbfs": True}):
        try:
            _config(setting)
        except ValueError:
            pass
        else:
            raise AssertionError(setting)
    assert _config({"position": {"x": 1}, "runtime_path": ["vad"]}) == DEFAULTS
    node = BLOCK.build_node_payload(node_id="vad", title="<script>bad</script>")
    for render in (BLOCK.render_modal, BLOCK.render_inspector_panel, BLOCK.render_node_card):
        html = render(node=node)["html"]
        assert "{{" not in html and "<script>" not in html, html
    for surface in ("modal", "inspector_panel"):
        assert BLOCK.model["ui_assets"][surface], "Package assets must be declared statically"
        for asset in BLOCK.model["ui_assets"][surface]:
            assert (BLOCK.directory / asset["path"]).is_file()
    assert (BLOCK.directory / "README.md").is_file()
    result = BLOCK.handle_ui_action(node=node, action="save_properties",
        values={"title": "Voix", "config": {"silence_ms": "600"}})
    assert result["node_patch"]["config"]["silence_ms"] == 600
    legacy = _config({"speech_start_ms": 60})
    assert legacy["speech_start_ms"] == 60 and legacy["min_level_dbfs"] == -48
    assert "error" in BLOCK.handle_ui_action(node=node, action="save_properties", values={"title": ""})


def test_port_order():
    """FB1/FB2: reordered inputs preserve lifecycle routing; malformed identities still fail."""
    for mode in ("centralized", "zeromq_active"):
        base = context(mode)
        for ports in (base.input_ports, base.input_ports[::-1]):
            received = []
            ctx = replace(context(mode, services={"runtime_listener": SimpleNamespace(send=received.append)}), input_ports=ports)
            before = [vars(p).copy() for p in ports]
            assert BLOCK.prepare_runtime(ctx).listen_on_run == (mode == "zeromq_active")
            ctx.input_attribute("command_in").update('{"action":"start","stream_id":"reordered"}')
            with patch("subprocess.Popen") as popen:
                result = BLOCK.execute_runtime(ctx)
                assert result.status == ("success" if mode == "zeromq_active" else "skipped"), result
                assert not result.outputs
                popen.assert_not_called()
            assert received == ([{"action": "start", "stream_id": "reordered"}] if mode == "zeromq_active" else [])
            assert [vars(p) for p in ctx.input_ports] == before
        invalid_ports = [{"input_ports": base.input_ports[:1]},
                         {"input_ports": (base.input_ports[0], base.input_ports[0])},
                         {"output_ports": ()}, {"output_ports": base.output_ports * 2}]
        for field, ports in (("input_ports", base.input_ports), ("output_ports", base.output_ports)):
            for index, port in enumerate(ports):
                changes = [{"id": 99}, {"name": "wrong"},
                           {"transport": "message" if getattr(port, "transport", "message") == "audio_stream" else "audio_stream"}]
                if field == "input_ports":
                    changes += [{"required": True}, {"multiplicity": "many"}, {"execution_requirement": "required_for_execution"}]
                for change in changes:
                    changed = list(ports)
                    changed[index] = SimpleNamespace(**{**vars(port), **change})
                    invalid_ports.append({field: tuple(reversed(changed))})
        for changes in invalid_ports:
            ctx = replace(context(mode), **changes)
            try:
                BLOCK.prepare_runtime(ctx)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid ports accepted: {changes}")
            assert BLOCK.execute_runtime(ctx).status == "failed"


def test_commands():
    """FB1/FB2/FB4: producer commands are explicit and stale values cannot replay lifecycle actions."""
    command = '{"action":"start","stream_id":"capture"}'
    received = []
    event = BlockInputEvent(edge_id="command", input_port_id=2, input_port_name="command_in",
        source_node_id="micro", source_port_id=2, value=command, content_type="application/json", sequence=1)
    ctx = context(input_events=(event,), services={"runtime_listener": SimpleNamespace(send=received.append)})
    assert BLOCK.execute_runtime(ctx).status == "success" and received == [json.loads(command)]
    stale = context(inputs={"command_in": command}, services=ctx.services)
    stale.mark_inputs_consumed()
    assert BLOCK.execute_runtime(stale).status == "skipped" and len(received) == 1
    for raw in ("null", "[]", '{"action":"interrupt"}', '{"action":"stop","stream_id":"x"}',
                '{"action":"start","stream_id":""}', '{"action":"start","stream_id":"x","extra":1}', " " * 4097):
        try:
            _command(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(raw)


def test_native_and_clock():
    """FB3: real Silero classifies generated speech/silence; deterministic 32ms hysteresis has exact offsets."""
    events = []
    detector = _Detector(DEFAULTS, "voice", events.append)
    for offset in range(0, len(pcm_speech()), 131):
        detector.feed(pcm_speech()[offset:offset + 131])
    detector.finish()
    assert [event["event"] for event in events] == ["speech_started", "speech_stopped"], events
    assert events[1]["reason"] == "silence"
    assert events[0]["utterance_id"] == events[1]["utterance_id"]
    assert len({event["event_id"] for event in events}) == 2
    assert events[0]["audio_start_ms"] < events[0]["audio_end_ms"] < events[1]["audio_end_ms"]
    assert detector.samples == len(pcm_speech()) // 2
    silent = []
    _Detector(DEFAULTS, "silent", silent.append).feed(bytes(32000))
    assert silent == []
    # A controlled classifier isolates hysteresis math; native speech is tested separately above.
    exact = []
    detector = _Detector(DEFAULTS, "clock", exact.append)
    labels = iter([0.95] * 5 + [0.05] * 16)
    detector.vad = SimpleNamespace(probability=lambda data: next(labels))
    signal = b"\xe8\x03\x18\xfc" * 256  # AC energy, not DC or digital silence.
    detector.feed(signal * 21)
    assert [(e["event"], e["audio_start_ms"], e["audio_end_ms"]) for e in exact] == [
        ("speech_started", 0, 160), ("speech_stopped", 0, 672)]
    tail = []
    detector = _Detector(DEFAULTS, "tail", tail.append)
    detector.vad = SimpleNamespace(probability=lambda data: 0.95)
    detector.feed(signal * 5 + signal[:160])
    detector.finish()
    assert tail[-1]["reason"] == "stream_ended" and tail[-1]["audio_end_ms"] == 165


def noise_pcm(amplitude=100, *, clicks=False, seconds=10):
    """Return reproducible quiet noise or 10ms clicks each second, with no speech."""
    rng = Random(1729)
    samples = array("h", (rng.randint(-amplitude, amplitude) if not clicks or i % 16000 < 160 else 0
                          for i in range(seconds * 16000)))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def test_false_positive_regression():
    """FB3: neural classification and energy gates reject real silence, noise, clicks and DC bias."""
    for duration in (60, DEFAULTS["speech_start_ms"]):
        for label, pcm in (("silence", bytes(320000)), ("quiet noise", noise_pcm()),
                           ("small clicks", noise_pcm(clicks=True)),
                           ("loud clicks", noise_pcm(12000, clicks=True)),
                           ("DC bias", b"\xe8\x03" * 160000)):
            events = []
            detector = _Detector({**DEFAULTS, "speech_start_ms": duration}, "noise", events.append)
            for offset in range(0, len(pcm), 1973):
                detector.feed(pcm[offset:offset + 1973])
            detector.finish()
            assert not events, (duration, label, events)
    # Native classification remains active: speech following noise is not muted forever.
    events = []
    detector = _Detector(DEFAULTS, "recovery", events.append)
    detector.feed(noise_pcm(clicks=True))
    detector.feed(pcm_speech())
    detector.finish()
    assert [e["event"] for e in events] == ["speech_started", "speech_stopped"], events
    assert events[0]["audio_start_ms"] >= 10000


def test_speech_not_sound_regression():
    """FB3: reproduce intermittent noises that fooled WebRTC; retain brief and attenuated speech."""
    rng = Random(7)
    cases = [("intermittent white noise", array("h", (rng.randint(-800, 800) if i % 16000 < 6400 else 0
                                                     for i in range(96000))))]
    for frequency in (200, 440, 1000):
        cases.append((f"{frequency}Hz non-speech tone", array("h", (
            int(800 * math.sin(2 * math.pi * frequency * i / 16000)) if i % 16000 < 6400 else 0
            for i in range(96000)))))
    for name, data in cases:
        if sys.byteorder != "little":
            data.byteswap()
        events = []
        detector = _Detector(DEFAULTS, name, events.append)
        detector.feed(data.tobytes())
        detector.finish()
        assert not events, (name, events)
    for phrase in ("yes", "no", "Please stop the answer now"):
        for gain in (1.0, 0.2):
            samples = array("h", pcm_speech(phrase))
            if sys.byteorder != "little":
                samples.byteswap()
            scaled = array("h", (int(value * gain) for value in samples))
            if sys.byteorder != "little":
                scaled.byteswap()
            events = []
            detector = _Detector(DEFAULTS, phrase, events.append)
            detector.feed(scaled.tobytes())
            detector.finish()
            assert [e["event"] for e in events] == ["speech_started", "speech_stopped"], (phrase, gain, events)
            measure = events[0]["detection"]
            assert measure["engine"] == "silero_v6.2.1_onnx"
            assert .7 <= measure["speech_score"] <= 1 and math.isfinite(measure["level_dbfs"])
            assert events[0]["audio_end_ms"] - events[0]["audio_start_ms"] == 160


def test_hysteresis_diagnostics_and_model_integrity():
    """FB2/FB3/FB5: no noise-triggered outputs, bounded numeric diagnostics and no model fallback."""
    from blocs.voice_activity_detection.neural_vad import verified_model
    events, diagnostics = [], []
    detector = _Detector(DEFAULTS, "diagnostic", events.append, diagnostics.append)
    detector.last_diagnostic_at -= 6
    detector.feed(bytes(32768))
    assert not events and len(diagnostics) == 1
    assert diagnostics[0]["level_dbfs"] == -120 and "audio" not in diagnostics[0]
    signal = b"\xe8\x03\x18\xfc" * 256
    labels = iter([.8] * 5 + [.6] * 20 + [.1] * 16)
    detector.vad = SimpleNamespace(probability=lambda data: next(labels))
    detector.feed(signal * 41)
    detector.finish()
    assert [e["event"] for e in events] == ["speech_started", "speech_stopped"]
    assert len(diagnostics) == 1, "Score measurements must not flood logs or the graph."
    with TemporaryDirectory(prefix="vad-model-integrity-") as directory:
        missing = Path(directory) / "weights.onnx"
        for data in (None, b"untrusted weights"):
            if data is not None:
                missing.write_bytes(data)
            try:
                verified_model(missing)
            except ValueError:
                pass
            else:
                raise AssertionError("Missing/altered weights cannot trigger a silent WebRTC fallback.")
    with patch("blocs.voice_activity_detection.block.SileroSpeech", side_effect=AssertionError("No model in simulation")):
        assert BLOCK.initialize_runtime(context("centralized")).status == "success"
        assert BLOCK.execute_runtime(context("centralized")).status == "skipped"


def test_continuous_decoders():
    """FB3/FB4: each native WebM/Ogg decoder emits voice events before stop and drains exact PCM."""
    for container in ("webm", "ogg"):
        for channels in (1, 2):
            events = []
            capture = _Capture(DEFAULTS, "capture", events.append)
            data = encoded(container, channels)
            try:
                # Split even the container signature; no chunk is treated as an independent file.
                chunks = [data[:1], data[1:3], *[data[offset:offset + 997] for offset in range(3, len(data), 997)]]
                for sequence, chunk in enumerate(chunks, 1):
                    capture.feed(frame(chunk, sequence, channels=channels))
                    capture.pump()
                until(lambda: (capture.pump(), len(events) >= 2)[1], "Speech must be detected before producer stop.")
                assert [item["event"] for item in events] == ["speech_started", "speech_stopped"], events
                processed, previous = capture.detector.samples, list(events)
                # Missing transport data cannot advance the audio clock or manufacture a speech transition.
                for _ in range(20):
                    capture.pump()
                assert events == previous and capture.detector.samples >= processed
                capture.finish({"action": "stop", "stream_id": "capture", "frame_count": capture.frames,
                                "byte_count": capture.bytes, "aborted": False})
                until(capture.pump, "Explicit stop must drain and reap the decoder.")
                assert abs(capture.detector.samples - len(pcm_speech()) // 2) <= 1
                assert len(events) == 2, "A silence-ended utterance cannot end a second time on stream stop."
            finally:
                capture.close()
            assert capture.process.poll() is not None
            # Lossy codec ringing must not turn isolated clicks into durable speech.
            quiet_events = []
            quiet = _Capture(DEFAULTS, "clicks", quiet_events.append)
            try:
                data = encoded(container, channels, noise=True)
                quiet.feed(frame(data, stream_id="clicks", channels=channels))
                quiet.finish({"action": "stop", "stream_id": "clicks", "frame_count": 1,
                              "byte_count": len(data), "aborted": False})
                until(quiet.pump, "Encoded clicks must drain through the real decoder.")
                assert not quiet_events, (container, channels, quiet_events)
            finally:
                quiet.close()


def test_errors_and_cleanup():
    """FB4: gaps, abort, invalid headers, final totals and cancellation fail without false speech endings."""
    for cause in ("gap", "totals", "aborted", "header", "missing"):
        events = []
        capture = _Capture({**DEFAULTS, "drain_timeout_sec": 1}, "capture", events.append)
        try:
            if cause == "header":
                capture.feed(frame(b"Not an audio container"))
                operation = capture.pump
            else:
                capture.feed(frame(encoded()[:100]))
                if cause == "gap":
                    operation = lambda: capture.feed(frame(b"more", 3))
                elif cause == "missing":
                    capture.finish({"action": "stop", "stream_id": "capture", "frame_count": 2, "byte_count": 200, "aborted": False})
                    capture.stopped_at -= 2
                    operation = capture.pump
                else:
                    operation = lambda: capture.finish({"action": "stop", "stream_id": "capture",
                        "frame_count": 0, "byte_count": 0, "aborted": cause == "aborted"})
            try:
                operation()
            except ValueError:
                pass
            else:
                raise AssertionError(cause)
            assert not events
        finally:
            capture.close()
    capture = _Capture(DEFAULTS, "cancel", lambda event: None)
    capture.feed(frame(encoded()[:100], stream_id="cancel"))
    capture.pump()
    began = time.monotonic()
    capture.close()
    assert time.monotonic() - began < .5 and capture.process.poll() is not None


def test_real_graph_modes():
    """FB1–FB4: actual Microphone→VAD→Display routes survive reversed VAD inputs in both modes."""
    micro = MicrophoneStreamBlock()
    nodes = [micro.build_node_payload(node_id="micro"), BLOCK.build_node_payload(node_id="vad"),
             get_block_definition("display").build_node_payload(node_id="display")]
    nodes[1]["inputs"].reverse()
    graph = WorkflowGraph.from_payload({"nodes": nodes, "edges": [
        {"id": "audio", "fromNodeId": "micro", "fromPortId": 1, "toNodeId": "vad", "toPortId": 1},
        {"id": "commands", "fromNodeId": "micro", "fromPortId": 2, "toNodeId": "vad", "toPortId": 2},
        {"id": "events", "fromNodeId": "vad", "fromPortId": 1, "toNodeId": "display", "toPortId": 1}]})
    with TemporaryDirectory(prefix="vad-graph-") as directory:
        root = Path(directory)
        engine = WorkflowOrchestrator(root_dir=root, runs_dir=root / "runs", active_worker_host="thread")
        run = engine.prepare_active_run(graph)
        assert run.status == "prepared", run.logs
        try:
            session = engine._active_sessions[run.run_id]
            assert not session.controller.health_snapshot()["played"]
            source = session.controller.runtime_audio_stream_service.client_for("micro",
                port_routes=run.plan.worker_configs["micro"].runtime_audio_stream_port_routes)
            data = encoded("ogg", 2)
            # No start command is needed for detection: audio alone starts the listener's capture.
            source.publish_port("audio_out", data, codec="opus", sample_rate_hz=48000, channels=2, stream_id="graph-capture")
            until(lambda: run.results.get("vad", {}).get(BLOCK.kind, {}).get("event") == "speech_stopped",
                  f"VAD events must be published without Play: {run.logs}")
            until(lambda: run.node_statuses.get("display") == "success", "Speech JSON must execute a real downstream block.")
            action = micro.handle_ui_action(node=nodes[0], action="publish_capture_command", values={
                "action": "stop", "stream_id": "graph-capture", "frame_count": 1, "byte_count": len(data), "aborted": False
            })["active_runtime_actions"][0]
            engine.control_active_run(run.run_id, action="publish_output", node_id="micro",
                                      payload={key: value for key, value in action.items() if key not in {"action", "node_id"}})
            until(lambda: run.runtime_attributes.get("vad", {}).get("inputs", {}).get("command_in", {}).get("status") == "consumed",
                  "Optional lifecycle must traverse the real command graph.")
            received = run.runtime_attributes["vad"]["inputs"]["command_in"]
            assert received["source_node_id"] == "micro" and json.loads(received["value"])["stream_id"] == "graph-capture"
        finally:
            engine.stop_active_run(run.run_id)
        assert not any("shutdown_timeout" in line for line in run.logs), run.logs
        simulation = engine.create_run(graph, runtime_mode="centralized", auto_start=False)
        engine._execute_run(simulation)
        assert simulation.status == "success", simulation.logs
        assert not simulation.output_values.get("vad:1")


def test_properties(page, server, _errors, *, origin="managed"):
    """FB5: inspect real shell screenshots, contained diagnostics, narrow layouts and atomic Apply."""
    model = install_test_package(server, "voice_activity_detection", origin=origin)
    node = BLOCK.build_node_payload(node_id="vad", position={"x": 280, "y": 180})
    node["block_version"] = model["version"]
    for surface in ("modal", "inspector_panel"):
        surface_payload(server, model, node, surface)
    created = create_project_api(server, title="VAD properties", document=graph_payload("VAD properties", [node], []))["project"]
    graph_id = created.get("graph_id") or created["project_id"]
    page.goto(project_editor_url(server.base_url, graph_id, workspace_project_id=created["workspace_project_id"]))
    page.wait_for_selector('.canvas-node[data-node-id="vad"]')
    page.locator('.canvas-node[data-node-id="vad"] h3').dblclick()
    modal = page.locator('[data-generic-block-modal-root][data-node-kind="voice_activity_detection"]')
    modal.wait_for()
    for width, height, label in ((1440, 900, "desktop"), (390, 740, "mobile"), (320, 568, "small")):
        page.set_viewport_size({"width": width, "height": height})
        bounds = modal.evaluate("""panel => {
          const rect=panel.getBoundingClientRect(),body=panel.querySelector('.vad-body');
          const close=panel.querySelector('[data-close-block-modal]').getBoundingClientRect();
          const apply=panel.querySelector('[data-vad-apply]').getBoundingClientRect();
          return {background:getComputedStyle(panel).backgroundColor,left:rect.left,right:rect.right,
            overflow:body.scrollWidth>body.clientWidth+1,close:close.bottom<=innerHeight,apply:apply.bottom<=innerHeight};
        }""")
        assert bounds["background"] == "rgb(255, 255, 255)" and bounds["left"] >= 0 and bounds["right"] <= width, bounds
        assert not bounds["overflow"] and bounds["close"] and bounds["apply"], bounds
        assert modal.locator('[data-block-modal-error-panel]').count() == 1
        page.screenshot(path=artifact_path(f"vad-{origin}-modal-{label}.png"))
    setting = modal.locator('[data-vad-setting="silence_ms"]')
    setting.fill("0")
    modal.locator('[data-vad-apply]').click()
    assert "Vérifiez" in modal.locator('[data-vad-feedback]').inner_text()
    setting.fill("600")
    modal.locator('[data-vad-setting="min_level_dbfs"]').fill("-44")
    modal.locator('[data-vad-apply]').click()
    page.wait_for_function("document.querySelector('[data-generic-block-modal-root] [data-vad-feedback]')?.textContent.includes('Enregistré')")
    modal.locator('[data-close-block-modal]').click()
    page.set_viewport_size({"width": 1440, "height": 900})
    page.evaluate("openInspectorPanel()")
    page.locator('.right-rail').hover(position={"x": 20, "y": 100})
    page.locator('#pinInspectorButton').click()
    inspector = page.locator('#blockOwnedInspectorView [data-block-inspector-root][data-node-id="vad"]')
    inspector.wait_for(state="visible")
    expect_ui(inspector.locator('[data-vad-setting="silence_ms"]')).to_have_value("600", timeout=10000)
    expect_ui(inspector.locator('[data-vad-setting="min_level_dbfs"]')).to_have_value("-44", timeout=10000)
    assert not inspector.evaluate("element => element.scrollWidth > element.clientWidth + 1")
    page.screenshot(path=artifact_path(f"vad-{origin}-inspector.png"))


if __name__ == "__main__":
    for test in (test_contracts, test_port_order, test_commands, test_native_and_clock, test_false_positive_regression,
                 test_speech_not_sound_regression, test_hysteresis_diagnostics_and_model_integrity, test_continuous_decoders,
                 test_errors_and_cleanup, test_real_graph_modes):
        test()
        print(f"[ok] {test.__name__}", flush=True)
    for origin in ("managed", "linked"):
        run_playwright_smoke("F5.51_voice_activity_detection",
                            lambda page, server, errors: test_properties(page, server, errors, origin=origin))
