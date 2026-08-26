// A finding linked from the Analysis page. Findings default to status=open, so
// a link to a dismissed finding would land on a list that doesn't contain it —
// the deep link therefore clears the status filter before loading.
function requestedFindingId() {
  const id = new URLSearchParams(window.location.search).get("finding");
  return id ? Number(id) : null;
}

function currentFilters() {
  const form = document.getElementById("filter-form");
  const data = new FormData(form);
  const params = new URLSearchParams();
  for (const key of ["status", "category", "severity"]) {
    const val = data.get(key);
    if (val) params.set(key, val);
  }
  return params.toString();
}

function findingCard(f) {
  const dismissed = f.status === "dismissed";
  // A completed investigation that ruled its candidate out. It is a Stage 2
  // result like any other, but it is a record of work done rather than a risk:
  // there is nothing to dismiss, and dismissing it would write suppression
  // feedback for a pattern nobody objected to.
  const noIssue = f.status === "no_issue";
  // A hypothesis kept from an investigation the cost/turn budget cut short. It
  // keeps its evidence and stays dismissible — it's a claimed risk like any
  // other, just one nobody finished checking — but it must never be presented
  // as a settled finding, hence the label and the caveat below the title.
  const partial = f.status === "partial";
  return `
    <div class="card" data-id="${f.id}">
      <div class="meta">
        <span class="badge ${f.status}">${partial ? "unconfirmed" : f.status}</span>
        &nbsp;${escapeHtml(f.category)}${noIssue ? "" : ` &middot; severity: ${escapeHtml(f.severity)}`}
        ${f.node ? `&middot; node: ${escapeHtml(f.node)}` : ""}
        ${f.confidence != null ? `&middot; confidence: ${(f.confidence * 100).toFixed(0)}%` : ""}
        &middot; ${fmtTime(f.created_at)}
      </div>
      <strong>${escapeHtml(noIssue ? "No issue found — investigated and ruled out" : f.title)}${
        partial ? " (unconfirmed)" : ""
      }</strong>
      <p class="desc">${escapeHtml(f.description)}</p>
      ${f.recommendation ? `<p class="desc"><em>Recommendation:</em> ${escapeHtml(f.recommendation)}</p>` : ""}

      ${
        noIssue
          ? ""
          : `
      <details data-role="evidence">
        <summary>Show supporting events</summary>
        <div class="evidence" data-loaded="0">Loading on expand...</div>
      </details>`
      }

      <div class="investigation" data-loaded="0"></div>

      ${
        dismissed || noIssue
          ? ""
          : `
      <form class="dismiss-form inline" style="margin-top:0.75rem">
        <select name="scope">
          <option value="node">dismiss on this node only</option>
          <option value="global">dismiss on all nodes</option>
        </select>
        <input type="text" name="reason" placeholder="reason (optional)" style="flex:1; min-width: 180px" />
        <button type="submit" class="danger">Not important</button>
      </form>`
      }
    </div>
  `;
}

async function loadFindings() {
  const list = document.getElementById("findings-list");
  const query = currentFilters();
  const findings = await apiGet(`/api/findings${query ? "?" + query : ""}`);
  if (!findings.length) {
    list.innerHTML = `<div class="panel status-line">No findings match these filters.</div>`;
    return;
  }
  list.innerHTML = findings.map(findingCard).join("");

  // The trace and its cost come from the per-finding detail endpoint, so load
  // them per card rather than making the list endpoint N+1 queries deep.
  findings.forEach((f) => loadInvestigation(f.id));

  list.querySelectorAll("details").forEach((details) => {
    details.addEventListener(
      "toggle",
      async function handler() {
        if (!details.open) return;
        const card = details.closest(".card");
        const id = card.dataset.id;
        const evidenceDiv = details.querySelector(".evidence");
        if (evidenceDiv.dataset.loaded === "1") return;
        const detail = await apiGet(`/api/findings/${id}`);
        evidenceDiv.innerHTML = `
          <table>
            <thead><tr><th>Time</th><th>Node</th><th>Event</th><th>Severity</th><th>Message</th></tr></thead>
            <tbody>
              ${detail.evidence
                .map(
                  (e) => `<tr>
                    <td>${fmtTime(e.event_time)}</td>
                    <td>${escapeHtml(e.node)}</td>
                    <td>${escapeHtml(e.event_name)}</td>
                    <td>${escapeHtml(e.severity)}</td>
                    <td>${escapeHtml(e.message)}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>
        `;
        evidenceDiv.dataset.loaded = "1";
      },
      { once: false }
    );
  });

  list.querySelectorAll(".dismiss-form").forEach((form) => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const card = form.closest(".card");
      const id = card.dataset.id;
      const data = new FormData(form);
      try {
        await apiPost(`/api/findings/${id}/dismiss`, {
          scope: data.get("scope"),
          reason: data.get("reason") || null,
        });
        await loadFindings();
      } catch (err) {
        alert(err.message);
      }
    });
  });
}

// How this finding was reached: the agent's tool calls, what the investigation
// cost, and a way back to the run that produced it. Shown here rather than only
// on the Analysis page because this is where the finding is acted on.
async function loadInvestigation(findingId) {
  const card = document.querySelector(`.card[data-id="${findingId}"]`);
  if (!card) return;
  const div = card.querySelector(".investigation");
  if (!div || div.dataset.loaded === "1") return;
  div.dataset.loaded = "1";
  let detail;
  try {
    detail = await apiGet(`/api/findings/${findingId}`);
  } catch (err) {
    return;
  }
  const bits = [];
  if (detail.analysis_run_id != null) {
    const rank = detail.candidate_rank != null ? `, candidate rank ${detail.candidate_rank}` : "";
    bits.push(`from run #${detail.analysis_run_id}${rank}`);
  }
  if (detail.investigation_iterations != null) bits.push(`${detail.investigation_iterations} turns`);
  if (detail.investigation_cost_usd != null) bits.push(`~${fmtCost(detail.investigation_cost_usd)}`);

  div.innerHTML =
    (bits.length ? `<div class="status-line">Investigated: ${escapeHtml(bits.join(" · "))}</div>` : "") +
    traceSection(detail.steps, "How this was investigated");
}

function highlightRequestedFinding() {
  const id = requestedFindingId();
  if (!id) return;
  const card = document.querySelector(`.card[data-id="${id}"]`);
  if (!card) return;
  card.classList.add("highlight");
  card.scrollIntoView({ behavior: "smooth", block: "center" });
}

onSubmit("filter-form", () => loadFindings());

// A deep link must be able to reach a dismissed finding, which the default
// status=open filter would hide.
if (requestedFindingId()) {
  document.querySelector("#filter-form [name=status]").value = "";
}
loadFindings().then(highlightRequestedFinding);
