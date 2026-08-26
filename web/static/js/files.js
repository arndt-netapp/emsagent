async function refreshFiles() {
  const files = await apiGet("/api/files");
  const body = document.getElementById("files-body");
  body.innerHTML = files
    .map(
      (f) => `
      <tr>
        <td>${f.id}</td>
        <td>${escapeHtml(f.filename)}</td>
        <td>${f.cluster ? escapeHtml(f.cluster) : '<span class="muted-cell">unspecified</span>'}</td>
        <td><span class="badge ${f.status}">${f.status}</span></td>
        <td>${escapeHtml(f.detected_format || "-")}</td>
        <td>${
          f.severity_filter
            ? `${escapeHtml(f.severity_filter)}+`
            : '<span class="muted-cell" title="Ingested with no severity floor.">all</span>'
        }${
          f.severity_skipped
            ? `<span class="dup-note" title="Parsed from this file but below its severity floor, so not stored.">-${f.severity_skipped} low</span>`
            : ""
        }</td>
        <td>${f.event_count}${
          f.duplicates_skipped
            ? `<span class="dup-note" title="Already ingested by an earlier fetch of this cluster, so not stored again.">+${f.duplicates_skipped} dup</span>`
            : ""
        }</td>
        <td>${fmtTime(f.discovered_at)}</td>
        <td>${
          f.status === "processed" && f.event_count
            ? `<a class="analyze-link" href="/analysis.html?file=${f.id}">Analyze</a>`
            : ""
        }</td>
      </tr>
      ${f.error_message ? `<tr><td></td><td colspan="8" class="error-text">${escapeHtml(f.error_message)}</td></tr>` : ""}
    `
    )
    .join("");
}

onSubmit("upload-form", async (e) => {
  const input = document.getElementById("upload-input");
  const clusterInput = document.getElementById("upload-cluster");
  const severityInput = document.getElementById("upload-min-severity");
  const status = document.getElementById("upload-status");
  if (!input.files.length) return;
  status.textContent = "Uploading...";
  try {
    const file = await apiUpload("/api/files/upload", input.files[0], {
      cluster: clusterInput.value,
      min_severity: severityInput.value,
    });
    const where = file.cluster ? ` for cluster ${file.cluster}` : " (cluster unspecified)";
    const dropped = file.severity_skipped
      ? ` ${file.severity_skipped} below ${file.severity_filter} were skipped.`
      : "";
    status.textContent = `Ingested ${file.event_count} events${where}.${dropped}`;
    input.value = "";
    clusterInput.value = "";
    await refreshFiles();
  } catch (err) {
    status.textContent = err.message;
  }
});

onSubmit("cluster-form", async (e) => {
  const status = document.getElementById("cluster-status");
  const form = new FormData(e.target);
  const body = {
    cluster: form.get("cluster"),
    username: form.get("username"),
    password: form.get("password"),
    count: parseInt(form.get("count"), 10) || 500,
    // "all" and null both mean no floor; the select always sends one of the
    // option values, so this is never an empty string.
    min_severity: form.get("min_severity") || "all",
    log_message: form.get("log_message") || null,
    verify_tls: form.get("verify_tls") === "on",
  };
  status.textContent = "Fetching from cluster...";
  try {
    const file = await apiPost("/api/clusters/fetch-events", body);
    status.textContent = file.duplicates_skipped
      ? `Fetched ${file.event_count + file.duplicates_skipped} events -> ${file.filename}: ${file.event_count} new, ${file.duplicates_skipped} already ingested.`
      : `Fetched ${file.event_count} events -> ${file.filename}`;
    e.target.reset();
    await refreshFiles();
  } catch (err) {
    status.textContent = err.message;
  }
});

refreshFiles();
