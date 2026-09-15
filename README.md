# Voice Activity Detection (VAD)

<!-- block-metadata:start -->
[![Block version: 0.1.1](https://img.shields.io/badge/block-0.1.1-blue)](model.json)
[![BloxSmith compatibility: 1.0.9](https://img.shields.io/badge/BloxSmith-1.0.9-brightgreen)](compatibility.json)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Verified BloxSmith versions: **1.0.9** (bundled-block tests; see [test evidence](compatibility.json)).
<!-- block-metadata:end -->


Locally detects periods of speech using **Silero VAD v6.2.1 (ONNX)**, without transcribing, identifying speakers, recording files or sending audio to an external service. The block emits JSON events; an existing Python block can decide which commands to send to other blocks. VAD itself does not stop any block.

## Wiring

- `Microphone Stream.audio_out → audio_in`: Opus in WebM/Ogg, 48 kHz clock, mono or stereo, one incoming audio link.
- `Microphone Stream.command_out → command_in`: **optional but recommended** JSON link to close the decoder cleanly after each capture. The equivalent TTS ports are also compatible.
- `events_out → Python`: JSON, with multiple subscribers supported.

Inputs `audio_in` (ID 1) and `command_in` (ID 2) may be reordered visually. Detection and commands follow port IDs without reordering the graph. Names, transports and multiplicities remain fixed; missing, duplicate or incompatible ports are rejected.

**Run prepares listening; Play and a start command are not required.** The first audio chunk creates its decoder. An optional start may arrive before or after the first chunks. Audio and commands always travel on separate links.

`command_in` recognizes producer commands:

```json
{"action":"start","stream_id":"unique-capture"}
{"action":"stop","stream_id":"unique-capture","frame_count":12,"byte_count":32000,"aborted":false}
```

Stop waits for frame/byte totals, closes stdin, drains the decoder and finishes the capture. This reconciliation allows stop to precede the final frames. An aborted stop, data loss, an error or runtime Stop **does not fabricate** a successful speech-end event. Retired IDs cannot be reused within the Run.

Without a command link, detection works while audio flows; missing frames are neither silence nor proof that a stream ended. The decoder waits for more data until runtime Stop. At most four captures may be open: connect commands when recording successive captures.

## Events

```json
{
  "event": "speech_started",
  "reason": "voice",
  "stream_id": "unique-capture",
  "utterance_id": "unique-utterance-id",
  "event_id": "unique-event-id",
  "audio_start_ms": 544,
  "audio_end_ms": 704,
  "detection": {"engine": "silero_v6.2.1_onnx", "speech_score": 0.94,
    "level_dbfs": -30.2, "speech_threshold": 0.7, "release_threshold": 0.55,
    "confirmation_ms": 160, "audio_processed_ms": 704}
}
```

- `speech_started`: a sufficient Silero speech score for at least `speech_start_ms`, with sufficient actual signal level in every window. Loud noise alone is not enough.
- `speech_stopped`, `reason: "silence"`: enough PCM classified as non-speech.
- `speech_stopped`, `reason: "stream_ended"`: an active utterance ends when the producer closes with verified totals. No end event is emitted without a matching start.

A start/end pair shares its `utterance_id`; each event has its own `event_id`. Offsets restart at zero for every `stream_id`. `audio_start_ms` is the beginning of the speech candidate, before confirmation. **`audio_end_ms` is the PCM actually processed at event time**, including the silence confirming an end. It is neither network arrival time nor the last voiced instant. STT can wait for this offset before closing its segment. `detection` reports actual measurements: the model score is not calibrated certainty or speaker identification. Existing JSON fields remain stable.

While idle, diagnostic metadata is refreshed at most every 5 seconds per capture, without graph output or per-frame logs. Start/end logs include score, level and threshold.

Example Python policy: on `event == "speech_started"`, publish `{"action":"interrupt"}` to destinations implementing that command. Actual browser playback interruption and agent cancellation depend on their own contracts; VAD does not add them to the framework.

## Settings

| Parameter | Default | Limits | Effect |
| --- | --- | --- | --- |
| `speech_start_ms` | 160 | 20–500 ms | Consecutive speech with sufficient energy before a start |
| `silence_ms` | 500 | 100–2,000 ms | Non-speech duration before an end |
| `min_level_dbfs` | −48 | −80 to −20 dBFS | Minimum RMS level after DC removal |
| `aggressiveness` | 2 | 0–3 | Silero onset thresholds: 0.50 / 0.60 / 0.70 / 0.80 |
| `drain_timeout_sec` | 5 | 1–20 s | Wait for final bytes/decoder after stop |

The block validates integer settings. **Apply** saves properties; **Stop, then Run** activates new settings. Controls and diagnostics use an opaque, internally scrolling panel with accessible Close/Apply actions.

VAD analyzes 32 ms PCM windows, mono 16 kHz **inside the block**. Each window must meet both the Silero speech threshold and the energy threshold. After confirmation, the continuation threshold is 0.15 lower than the onset threshold: this hysteresis avoids splitting speech when scores fluctuate slightly. Each capture has its own neural state; no state is shared between microphones or Runs.

This rejects very quiet noise and prevents a brief click followed by silence from counting as sustained speech. Microphone DC offset and padding of a final incomplete window do not inflate measured energy or confirm a new start. Durations round up to the next whole window. The emitted start offset remains that of the first candidate window, not its confirmation.

A threshold closer to zero (for example −40) rejects more quiet noise; a lower threshold (−55) accepts quieter speech. This safeguard is neither speaker recognition nor a guarantee against all sustained noise. Existing blueprints retain their ports and explicit settings; `aggressiveness` now controls Silero thresholds with no automatic WebRTC fallback. Existing durations, such as 60 ms, remain unchanged; a missing level threshold defaults to −48 dBFS. Consider setting older nodes to 160 ms.

Links remain 48 kHz Opus. Capture latency (250 ms by default in Microphone Stream), transport and decoding add to the analysis windows and hysteresis. A configured delay is not an end-to-end latency guarantee.

Aggressive filtering can miss quiet speech; music and noise may still be classified as speech. A detected end is not a sentence boundary. **Use headphones for the first trial**: VAD cannot distinguish your voice from Speaker output and does not provide echo cancellation. Real acoustic quality requires a hardware test.

## Installation and modes

FFmpeg, NumPy and ONNX Runtime must be available to the server's interpreter. From this block repository:

```sh
python3 -m pip install -r requirements.txt
```

Prefer the application's virtual environment. The block installs nothing during a Run. After adding this new block type, restart the application before inserting it into a blueprint: importing the package reloads the catalog, but document validation may retain the type list known at startup. The block neither patches the framework nor automatically restarts the application.

Official weights are bundled in `assets/models/`, with their MIT license and a SHA-256 checksum verified at load time. No Torch, GPU, remote service or runtime download is needed. A missing or altered model fails explicitly; there is no hidden degraded detector. See [model provenance](assets/models/README.md) and [Silero sources](https://github.com/snakers4/silero-vad).

- **Active Runtime**: a listener starts on Run, with at most four independent decoders.
- **Simulation**: `skipped`; no process, subscription, detection or fake event. Catalog and simulation remain usable without active-mode dependencies.
- Total compressed buffer: 8 MiB; frame limit: 512 KiB; retired IDs: 128. Pipes are non-blocking, stderr is drained and I/O pauses are bounded.
- FFmpeg accesses only pipes and decodes the expected Opus, with no audio path or network access. Discontinuities produce a diagnostic rather than classification of corrupted audio.
- State is local to the listener/run. Stop kills and reaps only block-owned processes, without claiming that all final events were delivered.

## Verification

### Release UI contract (0.1.1)

The manifest now declares modal and inspector assets, including their shared
relative import. Both entrypoints export ES-module `mount` functions and share a
release-local settings helper; there is no unversioned browser registration.
CSS is scoped to `voice_activity_detection@0.1.1`. Closing/replacing a surface
detaches its field listeners without touching the detector.

This migration does not change the Silero model, speech thresholds, audio ports
or false-positive filtering. Existing releases and blueprints are not rewritten:
install or reload `0.1.1` and explicitly select that node version. Legacy
unversioned UI loading is not supported by these release modules.

From the private `bloxmith-blocs` test workspace, run `python3 -B tests/run_tests.py voice_activity_detection`. Captures go to ignored test results, without hard-coded personal paths.

Suite `F5.51_voice_activity_detection.py` covers contracts, both modes, the real Silero model on speech synthesized locally using FFmpeg `flite`, silence, repeated non-voice clicks, intermittent 400 ms noise, non-speech tones, brief replies, attenuated speech, quiet noise, DC offset, speech after noise, fragmented windows, hysteresis, offsets, mono/stereo WebM/Ogg Opus, events before stop, clean shutdown, counter/stream errors, cancellation and a Microphone → VAD → Display graph.

Reversed input ordering is covered during preparation, command reception and the real mini-graph, in simulation and Active Runtime; invalid contracts remain rejected. Properties are exercised in the real shell: validation, saving, inspector, opaque modal, internal scrolling and accessible actions at 1,440, 390 and 320 px. The card uses public escaped-text rendering; titles are never interpreted as HTML. `flite` is a test-only prerequisite. No remote API, microphone permission, key or user data is used.

The properties regression now installs and links the current package through the
real framework, checking returned assets and persisted settings in both cases.
`F5.52_vad_release_runtime.py` additionally checks simulation and feeds locally
synthesized Opus through the public browser bridge to the installed/linked VAD
before Play. A real downstream display must receive the speech event.

The package owns its model, Python implementation, templates, modal/inspector assets, README, dependencies and tests. Only public `bloxsmith_app.block_api` imports cross the framework boundary in `block.py`.

## Compatibility policy

[compatibility.json](compatibility.json) records HackInvent's verified BloxSmith versions and test evidence. The outer harness is a **bundled-block test installation**; the release-specific suites described above additionally install and link this package through the real framework. Evidence covers those explicit cases, not every distribution format, browser/OS or live provider. Other framework versions are unverified, not necessarily incompatible.

The block-version badge follows `model.json`, not a published Git tag. `unversioned` means that no block release version is declared; no number is inferred from the framework version. The framework still uses `model.json` for its runtime/install contract; the tester-owned JSON does not replace it. Official integration tests run in the private `bloxmith-blocs` workspace. Test helpers and the proprietary framework are not bundled in this public block repository.
