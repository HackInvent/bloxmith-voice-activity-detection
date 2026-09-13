/** Register block-owned VAD inspector settings. */
(function () {
  const registry = (window.CWBlockUiBlocks = window.CWBlockUiBlocks || {});
  registry.voice_activity_detectionInspectorPanel = {
    /** Bind inspector fields; its lifetime never owns audio or the detector. */
    mount(root, api) { return window.CWVoiceActivityDetection.mount(root, api); },
  };
})();
