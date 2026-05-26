const $ = (sel) => document.querySelector(sel);

const statusLine = $("#status-line");
const chatModel = $("#chat-model");
const chatMessages = $("#chat-messages");
const chatForm = $("#chat-form");
const chatInput = $("#chat-input");
const chatSend = $("#chat-send");
const streamToggle = $("#stream-toggle");
const compareModels = $("#compare-models");
const compareRun = $("#compare-run");
const compareStatus = $("#compare-status");
const compareResults = $("#compare-results");
const reportContent = $("#report-content");
const reportRefresh = $("#report-refresh");

let appStatus = null;

function setStatus(text, kind = "") {
  statusLine.textContent = text;
  statusLine.className = `subtitle ${kind}`;
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.id === `panel-${name}`);
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

function appendMessage(role, text, meta = "") {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  const pre = document.createElement("pre");
  pre.textContent = text;
  el.appendChild(pre);
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    el.appendChild(m);
  }
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function formatMeta(result) {
  const parts = [`${Math.round(result.total_latency_ms)} ms`];
  if (result.time_to_first_token_ms != null) {
    parts.push(`TTFT ${Math.round(result.time_to_first_token_ms)} ms`);
  }
  parts.push(`${result.tokens_per_second.toFixed(1)} tok/s`);
  return parts.join(" · ");
}

async function loadStatus() {
  try {
    const res = await fetch("/api/status");
    appStatus = await res.json();
    const mode = appStatus.mock ? "mock" : "live";
    const badge = appStatus.mock ? "MOCK" : "LIVE";
    const conn = appStatus.ollama_ok ? "Ollama ready" : "Ollama unreachable";
    setStatus(`${appStatus.hardware} · ${conn} · ${badge}`, mode);

    const models =
      appStatus.models?.length > 0 ? appStatus.models : appStatus.benchmark_models;
    chatModel.innerHTML = "";
    const defaultModel = appStatus.default_model;
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (m === defaultModel) opt.selected = true;
      chatModel.appendChild(opt);
    }
    if (!chatModel.value && models[0]) chatModel.value = models[0];

    compareModels.value = appStatus.benchmark_models.join(", ");
  } catch (err) {
    setStatus(`API error: ${err.message}`, "error");
  }
}

async function sendChat(prompt) {
  const model = chatModel.value;
  appendMessage("user", prompt);
  chatSend.disabled = true;

  const useStream = streamToggle.checked;

  if (!useStream) {
    const assistantEl = appendMessage("assistant", "…");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      assistantEl.querySelector("pre").textContent = data.response;
      assistantEl.querySelector(".meta")?.remove();
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = formatMeta(data);
      assistantEl.appendChild(meta);
    } catch (err) {
      assistantEl.querySelector("pre").textContent = `Error: ${err.message}`;
    } finally {
      chatSend.disabled = false;
    }
    return;
  }

  const assistantEl = appendMessage("assistant", "");
  const pre = assistantEl.querySelector("pre");

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, prompt }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || res.statusText);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const block of lines) {
        const line = block.trim();
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.type === "token") {
          pre.textContent += payload.text;
          chatMessages.scrollTop = chatMessages.scrollHeight;
        } else if (payload.type === "done") {
          const meta = document.createElement("div");
          meta.className = "meta";
          meta.textContent = formatMeta(payload.result);
          assistantEl.appendChild(meta);
        } else if (payload.type === "error") {
          pre.textContent = `Error: ${payload.message}`;
        }
      }
    }
  } catch (err) {
    pre.textContent = `Error: ${err.message}`;
  } finally {
    chatSend.disabled = false;
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = "";
  sendChat(text);
});

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

function compareTableHtml(report) {
  const qualityByModel = Object.fromEntries(report.quality.map((q) => [q.model, q]));
  let html = `<table>
    <thead><tr><th>Model</th><th>TTFT</th><th>Latency</th><th>tok/s</th><th>Quality</th></tr></thead><tbody>`;
  for (const b of report.benchmarks) {
    const q = qualityByModel[b.model];
    html += `<tr>
      <td><code>${escapeHtml(b.model)}</code></td>
      <td>${b.avg_ttft_ms != null ? Math.round(b.avg_ttft_ms) + " ms" : "n/a"}</td>
      <td>${Math.round(b.avg_latency_ms)} ms</td>
      <td>${b.avg_tokens_per_second.toFixed(1)}</td>
      <td>${q ? Math.round(q.score * 100) + "%" : "n/a"}</td>
    </tr>`;
  }
  html += "</tbody></table>";
  return html;
}

function renderCompareTable(report) {
  compareResults.innerHTML = compareTableHtml(report);
  compareResults.classList.remove("hidden");
}

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

compareRun.addEventListener("click", async () => {
  compareRun.disabled = true;
  compareStatus.className = "status-msg running";
  compareStatus.textContent = "Running benchmarks and quality checks — this may take a while…";
  compareResults.classList.add("hidden");

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models: compareModels.value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderCompareTable(data);
    compareStatus.className = "status-msg done";
    compareStatus.textContent = "Done — saved to data/results/latest.json";
    renderReportPreview(data);
  } catch (err) {
    compareStatus.className = "status-msg error";
    compareStatus.textContent = err.message;
  } finally {
    compareRun.disabled = false;
  }
});

function renderReportPreview(report) {
  const mode = report.mock
    ? '<span class="badge mock">Mock</span>'
    : '<span class="badge live">Live</span>';
  let html = `<p>${mode} Generated ${escapeHtml(report.generated_at)}</p>`;
  html += `<p><small>${escapeHtml(report.hardware_note)}</small></p>`;
  html += `<h3>Summary</h3>${compareTableHtml(report)}`;
  reportContent.innerHTML = html;
}

reportRefresh.addEventListener("click", async () => {
  reportContent.innerHTML = "<p class='hint'>Loading…</p>";
  try {
    const res = await fetch("/api/report/latest");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderReportPreview(data);
  } catch (err) {
    reportContent.innerHTML = `<p class="hint">${escapeHtml(err.message)}</p>`;
  }
});

loadStatus();
