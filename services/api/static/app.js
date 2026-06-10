const tenantInput = document.getElementById("tenant-id");
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");
const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const queryBtn = document.getElementById("query-btn");
const messages = document.getElementById("messages");

function tenantId() {
  return tenantInput.value.trim() || "default";
}

function showStatus(element, text, type) {
  element.hidden = false;
  element.textContent = text;
  element.className = `status ${type}`;
}

function appendMessage(role, text, meta) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "message-meta";
    metaEl.textContent = meta;
    div.appendChild(metaEl);
  }
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

async function parseError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return JSON.stringify(data.detail ?? data);
  } catch {
    return response.statusText || "Request failed";
  }
}

uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    showStatus(uploadStatus, "Choose a file first.", "error");
    return;
  }

  uploadBtn.disabled = true;
  uploadStatus.hidden = true;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("tenant_id", tenantId());

  try {
    const response = await fetch("/ingest", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    const result = await response.json();
    showStatus(
      uploadStatus,
      `Ingested ${result.child_count ?? "?"} chunks (${result.parent_count ?? "?"} parents) in ${(result.duration_seconds ?? 0).toFixed(1)}s.`,
      "success",
    );
    fileInput.value = "";
  } catch (err) {
    showStatus(uploadStatus, err.message || "Upload failed.", "error");
  } finally {
    uploadBtn.disabled = false;
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  appendMessage("user", query);
  queryInput.value = "";
  queryBtn.disabled = true;

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, tenant_id: tenantId() }),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    const result = await response.json();
    const sourceCount = result.sources?.length ?? 0;
    const meta = `${result.latency_ms}ms · ${sourceCount} sources${result.cache_hit ? " · cache hit" : ""}`;
    appendMessage("assistant", result.answer, meta);
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message || "Query failed."}`);
  } finally {
    queryBtn.disabled = false;
    queryInput.focus();
  }
});
