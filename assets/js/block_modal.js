/** Register block-owned VAD modal settings; never start recording from properties. */
(function () {
  const registry = (window.CWBlockUiBlocks = window.CWBlockUiBlocks || {});
  registry.voice_activity_detection = {
    /** Bind the modal through the public UI facade and return surface-only cleanup. */
    mount(root, api) { return window.CWVoiceActivityDetection.mount(root, api); },
  };
})();
