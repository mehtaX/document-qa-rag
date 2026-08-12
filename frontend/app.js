const uploadForm = document.getElementById("upload-form");
const queryForm = document.getElementById("query-form");
const pdfFile = document.getElementById("pdf-file");
const uploadBtn = document.getElementById("upload-btn");
const uploadMsg = document.getElementById("upload-msg");
const submitBtn = document.getElementById("submit-btn");
const answerEl = document.getElementById("answer");
const metadataEl = document.getElementById("metadata");
const resultEl = document.getElementById("result");
const contextSection = document.getElementById("context-section");
const contextChunks = document.getElementById("context-chunks");
const errorEl = document.getElementById("error");
const loadingEl = document.getElementById("loading");
const loadingMsg = document.getElementById("loading-msg");
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const docList = document.getElementById("doc-list");

async function refreshStatus() {
  try {
    const res = await fetch("/health");
    const data = await res.json();

    if (data.building) {
      statusBadge.className = "badge badge-warn";
      statusBadge.textContent = "Indexing";
      statusText.textContent = data.last_step || "Building index...";
    } else if (data.ready) {
      statusBadge.className = "badge badge-ok";
      statusBadge.textContent = "Ready";
      statusText.textContent = `${data.total_chunks} chunks indexed`;
    } else {
      statusBadge.className = "badge badge-warn";
      statusBadge.textContent = "No docs";
      statusText.textContent = "Upload a PDF to begin";
    }

    docList.innerHTML = "";
    if (data.documents && data.documents.length) {
      data.documents.forEach((doc) => {
        const li = document.createElement("li");
        li.textContent = `${doc.filename} — ${doc.pages} pages, ${doc.chunks} chunks`;
        docList.appendChild(li);
      });
    }

    if (data.build_error) {
      showError(data.build_error);
    }
  } catch (err) {
    statusBadge.className = "badge badge-err";
    statusBadge.textContent = "Offline";
    statusText.textContent = "Cannot reach server";
  }
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
  errorEl.textContent = "";
}

function renderContextChunks(sources) {
  contextChunks.innerHTML = "";
  if (!sources || !sources.length) {
    contextSection.classList.add("hidden");
    return;
  }

  sources.forEach((src, idx) => {
    const details = document.createElement("details");
    details.className = "chunk-panel";
    details.open = idx === 0;

    const summary = document.createElement("summary");
    summary.textContent = `${src.citation} — ${src.source_file}`;
    details.appendChild(summary);

    const pre = document.createElement("pre");
    pre.textContent = src.content;
    details.appendChild(pre);

    contextChunks.appendChild(details);
  });

  contextSection.classList.remove("hidden");
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  uploadMsg.textContent = "";

  const file = pdfFile.files[0];
  if (!file) {
    showError("Please select a PDF file.");
    return;
  }

  uploadBtn.disabled = true;
  uploadMsg.textContent = "Uploading and indexing...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.detail || "Upload failed.");
    }
    uploadMsg.textContent = payload.message;
    pdfFile.value = "";
    await refreshStatus();
  } catch (err) {
    showError(err.message);
  } finally {
    uploadBtn.disabled = false;
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultEl.classList.add("hidden");
  contextSection.classList.add("hidden");
  hideError();
  answerEl.textContent = "";
  metadataEl.innerHTML = "";

  const question = document.getElementById("question").value.trim();
  if (!question) {
    showError("Please enter a question.");
    return;
  }

  submitBtn.disabled = true;
  loadingEl.classList.remove("hidden");
  loadingMsg.textContent = "Retrieving chunks via hybrid search...";

  try {
    const res = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const payload = await res.json();
      throw new Error(payload.detail || "Query failed.");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sources = [];
    let requestId = "";

    resultEl.classList.remove("hidden");

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);

        if (event.type === "sources") {
          sources = event.sources || [];
          requestId = event.request_id || "";
          renderContextChunks(sources);
          loadingMsg.textContent = "Generating streamed answer...";
        } else if (event.type === "token") {
          answerEl.textContent += event.token;
        } else if (event.type === "done") {
          answerEl.textContent = event.answer || answerEl.textContent;
          metadataEl.innerHTML = `
            <div class="metadata-item"><strong>Latency:</strong> ${event.latency_ms.toFixed(0)} ms</div>
            <div class="metadata-item"><strong>Request ID:</strong> ${event.request_id || requestId}</div>
          `;
        } else if (event.type === "error") {
          throw new Error(event.message || "Stream error.");
        }
      }
    }
  } catch (err) {
    showError(err.message);
  } finally {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = false;
  }
});

refreshStatus();
setInterval(refreshStatus, 10000);
