#!/usr/bin/env python3
"""FB1/FB2/FB3/FB4: installed and linked VAD packages consume real routed Opus before Play."""

from pathlib import Path
import base64
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from playwright.sync_api import sync_playwright
from blocs.microphone_stream.block import MicrophoneStreamBlock
from blocs.voice_activity_detection.block import VoiceActivityDetectionBlock
from block_test_packages import install_test_package, prepare_release_run
from ui_smoke_common import (
    isolated_server, graph_payload, create_project_api, project_editor_url,
    display_node, data_edge, create_run_api, wait_for_run_predicate, stop_run_api, wait_for_run_terminal,
    attach_console_guards, assert_no_blocking_console_errors,
)


def test_package(origin, speech):
    """Publish a synthetic recording via the real public bridge; never request a microphone."""
    with isolated_server() as server, sync_playwright() as playwright:
        model = install_test_package(server, "voice_activity_detection", origin=origin)
        source = MicrophoneStreamBlock().build_node_payload(node_id="source")
        detector = VoiceActivityDetectionBlock().build_node_payload(node_id="detector")
        detector["block_version"] = model["version"]
        detector["inputs"].reverse()
        document = graph_payload("Packaged speech detection", [source, detector, display_node("sink", "Events", 600, 0)], [
            data_edge("audio", "source", 1, "detector", 1),
            data_edge("command", "source", 2, "detector", 2),
            data_edge("event", "detector", 1, "sink", 1),
        ])
        simulation = create_run_api(server, document, runtime_mode="centralized")
        run = wait_for_run_terminal(server, simulation["run_id"], timeout_sec=20)
        assert run["status"] == "success", run.get("logs")
        assert not run.get("output_values", {}).get("detector:1")
        project = create_project_api(server, document=document)["project"]
        graph_id = project["project_id"]
        prepared = prepare_release_run(server, graph_id, document)
        run_id = prepared["run_id"]
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = attach_console_guards(page)
            page.goto(project_editor_url(server.base_url, graph_id, workspace_project_id=project["workspace_project_id"]))
            page.wait_for_function("typeof window.CWBlockUi?.createRuntimeAudioStreamsApi === 'function'")
            scope = {"available": True, "workspaceProjectId": project["workspace_project_id"],
                     "graphId": graph_id, "instanceId": "1", "runId": run_id, "nodeId": "source"}
            page.evaluate("""async ({scope, source, speech}) => {
              const api=window.CWBlockUi.createRuntimeAudioStreamsApi({
                actions: {getRuntimeAudioStreamContext: () => scope}
              }, () => source);
              const publisher=await api.openOutput({outputPort:"audio_out",codec:"opus",sampleRateHz:48000,channels:2});
              const bytes=Uint8Array.from(atob(speech), character => character.charCodeAt(0));
              try {
                for (let offset=0; offset<bytes.length; offset+=4096) {
                  if (!publisher.sendFrame(bytes.slice(offset,offset+4096))) throw new Error("Audio fixture saturated");
                }
              } finally { publisher.close(); }
            }""", {"scope": scope, "source": source, "speech": speech})
            run = wait_for_run_predicate(server, run_id,
                lambda value: value.get("results", {}).get("detector", {}).get("voice_activity_detection", {}).get("event") == "speech_stopped"
                and value.get("node_statuses", {}).get("sink") == "success"
                and "speech_stopped" in str(value.get("output_values")),
                "Versioned VAD did not detect routed speech before Play", timeout_sec=30)
            assert run.get("node_statuses", {}).get("sink") == "success", run.get("logs")
            assert "speech_stopped" in str(run.get("output_values")), run.get("output_values")
            assert_no_blocking_console_errors(errors)
        finally:
            browser.close()
            stopped = stop_run_api(server, run_id)
        assert "shutdown_timeout" not in str(stopped.get("logs", []))
    print(f"[ok] {origin} VAD: centralized skip and Active Runtime Opus recognition", flush=True)


def main():
    """Reuse the block's local speech fixture for both installation modes."""
    fixtures = runpy.run_path(str(Path(__file__).with_name("F5.51_voice_activity_detection.py")))
    speech = base64.b64encode(fixtures["encoded"]("ogg", 2)).decode("ascii")
    for origin in ("managed", "linked"):
        test_package(origin, speech)
    print("[ok] F5.52_vad_release_runtime")


if __name__ == "__main__":
    main()
