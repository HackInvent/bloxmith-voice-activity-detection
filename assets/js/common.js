/** Own VAD draft settings and atomic saves, independently of listener lifetimes. */
const mounts = new WeakMap();
/** Bind one surface's settings through the public action facade; return idempotent cleanup. */
export function mountSettings(root, api) {
  mounts.get(root)?.();
  const title = root.querySelector("[data-vad-title]");
  const fields = Array.from(root.querySelectorAll("[data-vad-setting]"));
  const button = root.querySelector("[data-vad-apply]");
  const feedback = root.querySelector("[data-vad-feedback]");
  const snapshot = () => ({ title: title.value, config: Object.fromEntries(fields.map(field => [field.dataset.vadSetting, field.value])) });
  let saved = JSON.stringify(snapshot()), busy = false, disposed = false;
  /** Explain dirty, saved and read-only states without changing a running detector. */
  function refresh() {
    if (disposed) return;
    button.disabled = busy || saved === JSON.stringify(snapshot()) || Boolean(api.isReadOnly?.());
    button.textContent = busy ? "Application…" : "Appliquer";
  }
  function dirty() {
    if (disposed) return;
    feedback.textContent = saved === JSON.stringify(snapshot()) ? "Aucune modification." : "Modifications non appliquées.";
    feedback.dataset.error = "false";
    refresh();
  }
  /** Validate a single snapshot and keep edits typed during the request unsaved. */
  async function apply() {
    if (button.disabled || disposed) return;
    const invalid = [title, ...fields].find(field => !field.checkValidity());
    if (invalid) {
      invalid.reportValidity(); feedback.textContent = "Vérifiez le champ signalé."; feedback.dataset.error = "true"; return;
    }
    const values = snapshot();
    busy = true; refresh();
    try {
      const result = await api.applyAction("save_properties", values);
      if (result?.error) throw new Error(result.error);
      saved = JSON.stringify(values);
      if (!disposed) feedback.textContent = saved === JSON.stringify(snapshot()) ? "Enregistré. Stop puis Run pour activer." : "Enregistré ; des modifications restent à appliquer.";
    } catch (error) {
      if (!disposed) { feedback.textContent = error.message || "Enregistrement impossible."; feedback.dataset.error = "true"; }
    } finally { busy = false; refresh(); }
  }
  for (const field of [title, ...fields]) { field.addEventListener("input", dirty); field.addEventListener("change", dirty); }
  button.addEventListener("click", apply); refresh();
  const observer = new MutationObserver(() => { if (!root.isConnected) cleanup(); });
  observer.observe(document.body, { childList: true, subtree: true });
  function cleanup() {
    if (disposed) return;
    disposed = true;
    observer.disconnect();
    mounts.delete(root);
    for (const field of [title, ...fields]) { field.removeEventListener("input", dirty); field.removeEventListener("change", dirty); }
    button.removeEventListener("click", apply);
  }
  mounts.set(root, cleanup);
  return cleanup;
}
