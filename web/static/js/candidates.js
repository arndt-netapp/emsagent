let candidatePollHandles = {};

// Split rather than summed — see the matching note in analysis.js. Stage 2's
// growing-conversation cache genuinely pays off, so a healthy investigation
// shows a large cheap cache-read figure next to a small cache-write one.
function fmtInvestigationTokens(c) {
  const parts = [`${c.investigation_input_tokens} in`];
  if (c.investigation_cache_creation_input_tokens) {
    parts.push(`${c.investigation_cache_creation_input_tokens} cache-write`);
  }
  if (c.investigation_cache_read_input_tokens) {
    parts.push(`${c.investigation_cache_read_input_tokens} cache-read`);
  }
  return `${parts.join(" + ")} / ${c.investigation_output_tokens} out`;
}

function candidateCard(c) {
  const actionable = c.status === "pending";
  const investigating = c.status === "investigating";
  return `
    <div class="card" data-id="${c.id}">
      <div class="meta">
        <span class="badge ${c.status}">${c.status}</span>
        &middot; rank ${c.rank} &middot; ${escapeHtml(c.category)}
        ${c.node ? `&middot; node: ${escapeHtml(c.node)}` : ""}
        ${c.confidence != null ? `&middot; confidence: ${(c.confidence * 100).toFixed(0)}%` : ""}
      </div>
      <p class="desc">${escapeHtml(c.rationale)}</p>
      ${
        actionable
          ? `
      <div class="inline">
        <button class="investigate-btn" data-id="${c.id}">Investigate</button>
        <button class="discard-btn danger" data-id="${c.id}">Discard</button>
      </div>`
          : investigating
          ? `<div class="status-line" data-role="investigation-status">Investigating…</div>`
          : `
      <div class="status-line">
        ${
          c.investigation_input_tokens != null
            ? `${fmtInvestigationTokens(c)} tokens, ~${fmtCost(c.investigation_cost_usd)}`
            : c.status === "discarded"
            ? `Discarded${c.discard_reason ? `: ${escapeHtml(c.discard_reason)}` : ""}`
            : c.status === "auto_suppressed"
            ? "Skipped — already known (suppressed or an open finding matches)."
            : ""
        }
      </div>
      ${c.investigation_error ? `<div class="error-text">${escapeHtml(c.investigation_error)}</div>` : ""}
      <div class="findings-for-candidate" data-loaded="0"></div>`
      }
    </div>
  `;
}

async function loadCandidates(runId) {
  const panel = document.getElementById("candidates-panel");
  panel.innerHTML = `<h2>Candidates for run #${runId}</h2><div class="panel status-line">Loading...</div>`;
  const candidates = await apiGet(`/api/analysis/runs/${runId}/candidates`);
  if (!candidates.length) {
    panel.innerHTML = `<h2>Candidates for run #${runId}</h2><div class="panel status-line">No candidates were flagged.</div>`;
    return;
  }
  panel.innerHTML = `<h2>Candidates for run #${runId}</h2><div id="candidates-list">${candidates
    .map(candidateCard)
    .join("")}</div>`;
  attachCandidateHandlers(runId);

  for (const c of candidates) {
    if (c.status === "investigating") pollCandidate(runId, c.id);
    else if (c.status === "investigated") loadCandidateFindings(c.id);
  }
}

// What an investigation produced, from the Analysis page's point of view.
//
// This page shows what Stage 1 flagged and which candidates Stage 2 was run
// against — decisions and spend. It never renders a result or an agent trace:
// every Stage 2 outcome, whether it confirmed an issue or ruled one out, lives
// on the Findings page under one consistent heading. Splitting the trace across
// two pages with two different labels was the confusing part.
async function loadCandidateFindings(candidateId) {
  const card = document.querySelector(`.card[data-id="${candidateId}"]`);
  if (!card) return;
  const div = card.querySelector(".findings-for-candidate");
  if (!div || div.dataset.loaded === "1") return;
  const detail = await apiGet(`/api/candidates/${candidateId}`);
  div.dataset.loaded = "1";

  if (!detail.findings.length) {
    // No record at all: the loop ended before the agent had even a working
    // hypothesis to keep. Deliberately unattributed — this covers a budget
    // bound, a model that stopped talking, and an investigation that failed
    // outright (which also lands on status='investigated', with its error
    // rendered separately above). Naming one of those as the cause would be a
    // guess. Whatever the reason, it is NOT a ruling-out, and is correctly not
    // recorded as one; had the agent recorded a hypothesis, it would be linked
    // below as an unconfirmed result.
    div.innerHTML =
      `<div class="status-line">Investigation ended without recording a result.</div>`;
    return;
  }

  // Same tab, deliberately: anyone who wants a new one can cmd/ctrl-click, and
  // target="_blank" annoys more people than it helps.
  div.innerHTML = detail.findings
    .map((f) => {
      const label =
        f.status === "no_issue"
          ? "No issue found — investigated and ruled out"
          : f.status === "partial"
          ? `${escapeHtml(f.title)} (${escapeHtml(f.severity)}, unconfirmed — budget ran out)`
          : `${escapeHtml(f.title)} (${escapeHtml(f.severity)})`;
      return `<div class="status-line">&rarr; <a href="/findings.html?finding=${f.id}">${label}</a> — full result and agent trace on the Findings page</div>`;
    })
    .join("");
}

function attachCandidateHandlers(runId) {
  document.querySelectorAll(".investigate-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.id;
      btn.disabled = true;
      try {
        await apiPost(`/api/candidates/${id}/investigate`);
        await loadCandidates(runId);
      } catch (err) {
        alert(err.message);
        btn.disabled = false;
      }
    });
  });
  document.querySelectorAll(".discard-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.id;
      btn.disabled = true;
      try {
        await apiPost(`/api/candidates/${id}/discard`, { reason: null });
        await loadCandidates(runId);
      } catch (err) {
        alert(err.message);
        btn.disabled = false;
      }
    });
  });
}

function pollCandidate(runId, candidateId) {
  if (candidatePollHandles[candidateId]) clearInterval(candidatePollHandles[candidateId]);
  candidatePollHandles[candidateId] = setInterval(async () => {
    try {
      const candidate = await apiGet(`/api/candidates/${candidateId}`);
      if (candidate.status !== "investigating") {
        clearInterval(candidatePollHandles[candidateId]);
        delete candidatePollHandles[candidateId];
        await loadCandidates(runId);
      }
    } catch (err) {
      clearInterval(candidatePollHandles[candidateId]);
      delete candidatePollHandles[candidateId];
    }
  }, 2000);
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest(".view-candidates-btn");
  if (btn) loadCandidates(btn.dataset.runId);
});
