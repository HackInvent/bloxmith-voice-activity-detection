import { withProperties } from "./properties.js";

/** Bind this release's settings; surface cleanup never owns the audio detector. */
import { mountSettings } from "./common.js";

/** Mount one settings surface through its injected public API; return UI-only cleanup. */
function mountOwned(root, api) {
  return mountSettings(root, api);
}

/** Keep the block behavior and add properties-only accessibility. */
export function mount(root, ...args) {
  return withProperties(mountOwned).call(this, root, ...args);
}
