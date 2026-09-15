/** Bind this release's settings; surface cleanup never owns the audio detector. */
import { mountSettings } from "./common.js";

/** Mount one settings surface through its injected public API; return UI-only cleanup. */
export function mount(root, api) {
  return mountSettings(root, api);
}
