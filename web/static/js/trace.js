// Rendering for a Stage 2 investigation's audit trail. Shared by the Analysis
// page (where a refuted candidate's trace is the only record of what the agent
// checked) and the Findings page (where it's the evidence for a conclusion
// someone is being asked to act on).
// Args are rendered inline as k=v rather than raw JSON: the point of the trace
// is to be skimmable ("it looked up the event, then checked HA state"), and a
// wall of JSON braces defeats that.
function fmtStepArgs(args) {
  if (!args || typeof args !== "object") return "";
  const parts = Object.entries(args)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : v}`);
  return parts.length ? `(${escapeHtml(parts.join(", "))})` : "()";
}

function stepRow(s) {
  const duration = s.duration_ms != null ? `${s.duration_ms}ms` : "";
  const body = s.error
    ? `<span class="error-text">${escapeHtml(s.error)}</span>`
    : escapeHtml(s.result_summary || "");
  return `
    <li class="trace-step">
      <div class="trace-head">
        <span class="trace-tool">${escapeHtml(s.tool_name)}</span>
        <span class="trace-args">${fmtStepArgs(s.tool_args)}</span>
        <span class="trace-time">${duration}</span>
      </div>
      <div class="trace-result">${body}</div>
    </li>`;
}

function traceSection(steps, label) {
  if (!steps || !steps.length) return "";
  // Grouped by turn so the loop structure is visible — one model call, then
  // the tools it chose — which is the part that actually shows this is an
  // agent rather than a single scripted query.
  const byIteration = new Map();
  steps.forEach((s) => {
    if (!byIteration.has(s.iteration)) byIteration.set(s.iteration, []);
    byIteration.get(s.iteration).push(s);
  });
  const groups = [...byIteration.entries()]
    .map(
      ([iteration, group]) => `
      <li class="trace-turn">
        <div class="trace-turn-label">Turn ${iteration}</div>
        <ul class="trace-steps">${group.map(stepRow).join("")}</ul>
      </li>`
    )
    .join("");
  return `
    <details class="trace">
      <summary>${escapeHtml(label || "Agent trace")} — ${steps.length} tool call${steps.length === 1 ? "" : "s"}</summary>
      <ul class="trace-turns">${groups}</ul>
    </details>`;
}
