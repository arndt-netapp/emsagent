let pollHandle = null;
let runInFlight = false;

// Cache writes and cache reads are broken out separately rather than summed
// into one "cached" figure: a write costs 1.25x base input and a read costs
// 0.1x, so lumping them together gives no hint why the cost moved. Stage 1
// no longer caches at all, so both are normally 0 here — a non-zero write
// column means something reintroduced a cache breakpoint.
function fmtTokens(r) {
  const parts = [`${r.input_tokens} in`];
  if (r.cache_creation_input_tokens) parts.push(`${r.cache_creation_input_tokens} cache-write`);
  if (r.cache_read_input_tokens) parts.push(`${r.cache_read_input_tokens} cache-read`);
  return `${parts.join(" + ")} / ${r.output_tokens} out`;
}

function fmtScope(scope, eventsConsidered) {
  if (!scope) return `${eventsConsidered} events`;
  const nodes = scope.nodes.length ? scope.nodes.join(", ") : "-";
  const files = scope.files.length ? scope.files.join(", ") : "-";
  const range =
    scope.min_time && scope.max_time ? `${fmtTime(scope.min_time)} – ${fmtTime(scope.max_time)}` : "-";
  const title = `Files: ${files}\nTime range: ${range}`;
  return `<span title="${escapeHtml(title)}">${scope.total} events · ${escapeHtml(nodes)}</span>`;
}

// A compact "mode · cluster" line, with the full scope (including the fetched
// filename) in the tooltip. The full label is far too long for a cell in a
// nine-column table: the filename is one unbreakable token, and a token sets
// the column's min-content width, which pushed the whole table past its panel
// and left the View-candidates button floating outside the grey box.
const SCOPE_MODE_LABELS = { recent_pull: "Most recent pull", last_24h: "Last 24h", file: "Single file" };

function fmtRunScopeLabel(r) {
  if (!r.scope_mode) return "";
  // Looked up rather than a ternary: a two-way mapping silently mislabeled
  // every mode that wasn't recent_pull, so 'file' runs displayed as "Last 24h".
  const mode = SCOPE_MODE_LABELS[r.scope_mode] || r.scope_mode;
  const cluster = r.scope_cluster || "unspecified";
  return `<div class="scope-label" title="${escapeHtml(r.scope_label || "")}">${escapeHtml(mode)} &middot; ${escapeHtml(cluster)}</div>`;
}

async function refreshRuns() {
  const runs = await apiGet("/api/analysis/runs");
  const body = document.getElementById("runs-body");
  body.innerHTML = runs
    .map(
      (r) => `
      <tr>
        <td>${r.id}</td>
        <td><span class="badge ${r.status}">${r.status}</span></td>
        <td>${fmtTime(r.started_at)}</td>
        <td>${fmtScope(r.scope, r.events_considered)}${fmtRunScopeLabel(r)}</td>
        <td>${r.candidates_generated}</td>
        <td>${r.candidates_auto_suppressed}</td>
        <td>${fmtTokens(r)}</td>
        <td>${fmtCost(r.cost_usd)}</td>
        <td>${r.status === "completed" ? `<button class="view-candidates-btn" data-run-id="${r.id}">View candidates</button>` : ""}</td>
      </tr>
      ${r.error_message ? `<tr><td></td><td colspan="8" class="error-text">${escapeHtml(r.error_message)}</td></tr>` : ""}
    `
    )
    .join("");
  return runs;
}

// Shows what's currently ingested and available to analyze, and keeps the
// trigger button disabled when there's nothing to run on. This is advisory
// only — the backend is the source of truth and refuses a run (409) if
// nothing has changed since the last completed one, even if this check is
// stale (e.g. logs were ingested from another tab).
async function refreshIngestedScope() {
  const el = document.getElementById("ingested-scope");
  const btn = document.getElementById("trigger-btn");
  try {
    const scope = await apiGet("/api/analysis/scope");
    if (scope.total === 0) {
      el.textContent = "No EMS events ingested yet — upload and ingest logs on the Files tab first.";
      if (!runInFlight) btn.disabled = true;
    } else {
      const nodes = scope.nodes.length ? scope.nodes.join(", ") : "-";
      const range =
        scope.min_time && scope.max_time ? `${fmtTime(scope.min_time)} – ${fmtTime(scope.max_time)}` : "-";
      el.textContent = `Currently ingested: ${scope.total} events from ${scope.files.length} file(s) (nodes: ${nodes}), ${range}.`;
      if (!runInFlight) btn.disabled = false;
    }
    return scope;
  } catch (err) {
    el.textContent = `Could not load ingested scope: ${err.message}`;
    return null;
  }
}

// The run-scope selector. Options are enumerated by the backend — two per
// cluster present in the data — rather than composed here from a cluster
// picker plus a mode picker, so the user makes one choice and no combination
// can be selected that doesn't correspond to real events.
//
// Every option is single-cluster by construction: correlating events across
// unrelated clusters is meaningless, and ONTAP's <cluster>-01/-02 node naming
// means two clusters can even share node names.
// Arriving from a file's Analyze button on the Files page. The selector
// normally offers two options per cluster; this adds a third for that one file
// and selects it, rather than listing every file permanently — "most recent
// pull" cannot express an older bundle, but a dropdown of every file ever
// ingested is the option sprawl this UI is deliberately avoiding.
function requestedFileId() {
  const id = new URLSearchParams(window.location.search).get("file");
  return id ? Number(id) : null;
}

async function refreshScopeOptions() {
  const select = document.getElementById("scope-select");
  const hint = document.getElementById("scope-hint");
  const btn = document.getElementById("trigger-btn");
  try {
    // The file option is resolved by the backend too (?file_id=), not built
    // here: a scope's cluster label and its cost estimate should have exactly
    // one source, and this page used to duplicate both.
    const fileId = requestedFileId();
    const options = await apiGet(
      fileId ? `/api/analysis/scope-options?file_id=${fileId}` : "/api/analysis/scope-options"
    );
    if (!options.length) {
      select.innerHTML = "";
      hint.textContent = "";
      if (!runInFlight) btn.disabled = true;
      return [];
    }
    const previous = select.value;
    select.innerHTML = options
      .map((o, i) => {
        const text = `${SCOPE_MODE_LABELS[o.mode] || o.mode} — ${o.cluster_label} (${o.event_count} events)`;
        return `<option value="${i}">${escapeHtml(text)}</option>`;
      })
      .join("");
    // Preserve the selection across the refresh that follows every run, but a
    // file arrived at via Analyze is pre-selected on first load.
    if (previous && options[previous]) select.value = previous;
    else if (fileId && options[0] && options[0].mode === "file") select.value = "0";
    window.__scopeOptions = options;
    updateScopeHint();
    if (!runInFlight) btn.disabled = false;
    return options;
  } catch (err) {
    hint.textContent = `Could not load run scopes: ${err.message}`;
    return [];
  }
}

function selectedScope() {
  const select = document.getElementById("scope-select");
  const options = window.__scopeOptions || [];
  return options[select.value] || null;
}

function updateScopeHint() {
  const scope = selectedScope();
  if (!scope) {
    document.getElementById("scope-hint").textContent = "";
    return;
  }
  // Cost first, before the button is pressed. Stage 1 is a single call over
  // the whole scope, so a large bundle is a large bill and there is no cap to
  // catch it — see pricing.estimate_stage1_cost_usd. Explicitly hedged,
  // because how well the corpus compacts is data-dependent and the estimate
  // can be several times high on repetitive data.
  // Two decimals, not fmtCost's four: this is an extrapolation, and rendering
  // it to a hundredth of a cent would claim a precision it does not have.
  const cost =
    scope.estimated_cost_usd != null
      ? ` — Stage 1 will cost roughly ${
          scope.estimated_cost_usd < 0.01 ? "<$0.01" : `$${scope.estimated_cost_usd.toFixed(2)}`
        } (estimate)`
      : "";
  document.getElementById("scope-hint").textContent = `${scope.label}${cost}`;
}

function pollRun(runId) {
  const status = document.getElementById("current-status");
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(async () => {
    try {
      const run = await apiGet(`/api/analysis/runs/${runId}`);
      status.textContent =
        run.status === "running"
          ? `Run #${run.id}: ${run.status} (iteration ${run.iterations})`
          : `Run #${run.id}: ${run.status} — ${fmtTokens(run)} tokens, ~${fmtCost(run.cost_usd)}`;
      await refreshRuns();
      if (run.status !== "running") {
        clearInterval(pollHandle);
        pollHandle = null;
        runInFlight = false;
        await refreshIngestedScope();
        await refreshScopeOptions();
      }
    } catch (err) {
      status.textContent = err.message;
      clearInterval(pollHandle);
      pollHandle = null;
      runInFlight = false;
      document.getElementById("trigger-btn").disabled = false;
    }
  }, 2000);
}

document.getElementById("trigger-btn").addEventListener("click", async () => {
  const status = document.getElementById("current-status");
  const btn = document.getElementById("trigger-btn");
  btn.disabled = true;
  runInFlight = true;
  status.textContent = "Starting analysis run...";
  try {
    const scope = selectedScope();
    const result = await apiPost(
      "/api/analysis/runs",
      scope ? { mode: scope.mode, cluster: scope.cluster, file_id: scope.file_id ?? null } : {}
    );
    status.textContent = `Run #${result.run_id}: ${result.status}`;
    await refreshRuns();
    pollRun(result.run_id);
  } catch (err) {
    status.textContent = err.message;
    runInFlight = false;
    btn.disabled = false;
  }
});

document.getElementById("scope-select").addEventListener("change", updateScopeHint);

refreshRuns();
refreshIngestedScope();
refreshScopeOptions();
