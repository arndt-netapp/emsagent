async function apiRequest(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch (e) {
      // ignore, keep statusText
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

const apiGet = (path) => apiRequest("GET", path);
const apiPost = (path, body) => apiRequest("POST", path, body === undefined ? {} : body);

// `fields` carries anything the upload needs alongside the file itself (today:
// the optional cluster). Empty values are omitted rather than sent as "", so
// the server sees an absent field and applies its own fallback.
async function apiUpload(path, file, fields = {}) {
  const formData = new FormData();
  formData.append("file", file);
  for (const [key, value] of Object.entries(fields)) {
    if (value !== null && value !== undefined && String(value).trim() !== "") {
      formData.append(key, String(value).trim());
    }
  }
  const res = await fetch(path, { method: "POST", body: formData });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(`${res.status}: ${data.detail || res.statusText}`);
  }
  return res.json();
}

function fmtTime(iso) {
  if (!iso) return "-";
  return new Date(iso).toLocaleString();
}

function fmtCost(usd) {
  if (usd == null) return "-";
  return `$${usd.toFixed(4)}`;
}

// Escapes for BOTH text and attribute contexts, which is why it is an explicit
// character map rather than the usual textContent/innerHTML trick.
//
// That trick serializes a TEXT node, and the HTML serializer only escapes
// &, < and > there — quotes come back untouched. Two call sites interpolate
// this into an attribute (analysis.js renders filenames and cluster names into
// title="..."), and both those values come from user input: an upload named
// `x" onmouseover="..." .log` closed the attribute and executed. Escaping the
// quotes here fixes every attribute use at once, including future ones.
const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

function escapeHtml(str) {
  return String(str == null ? "" : str).replace(/[&<>"']/g, (ch) => HTML_ESCAPES[ch]);
}


// Attach a submit handler by element id, tolerating a missing element.
//
// `document.getElementById(x).addEventListener(...)` throws when x is absent,
// and that exception aborts the whole script — so every listener registered
// AFTER it silently never attaches, and the forms it was meant to control fall
// back to native browser submission. That is how a removed "Scan watch
// directory" form took the cluster-fetch handler down with it and put a
// password in the URL. A page should degrade to one broken control, not to
// none.
function onSubmit(id, handler) {
  const el = document.getElementById(id);
  if (!el) {
    console.warn(`onSubmit: no element #${id}`);
    return;
  }
  el.addEventListener("submit", (e) => {
    e.preventDefault();
    handler(e);
  });
}
