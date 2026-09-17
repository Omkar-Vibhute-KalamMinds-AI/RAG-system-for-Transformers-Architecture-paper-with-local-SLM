const els = {
  thread: document.getElementById("thread"),
  empty: document.getElementById("emptyState"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  composer: document.getElementById("composer"),
  runMeta: document.getElementById("runMeta"),
  statusPill: document.getElementById("statusPill"),
  statusLabel: document.getElementById("statusLabel"),
  factModel: document.getElementById("factModel"),
  factEmbed: document.getElementById("factEmbed"),
  factDevice: document.getElementById("factDevice"),
  temperature: document.getElementById("temperature"),
  maxTokens: document.getElementById("maxTokens"),
  topK: document.getElementById("topK"),
  minScore: document.getElementById("minScore"),
  doSample: document.getElementById("doSample"),
  useHistory: document.getElementById("useHistory"),
  summarize: document.getElementById("summarize"),
  ragControls: document.getElementById("ragControls"),
  modeHint: document.getElementById("modeHint"),
  tempVal: document.getElementById("tempVal"),
  tokVal: document.getElementById("tokVal"),
  kVal: document.getElementById("kVal"),
  scoreVal: document.getElementById("scoreVal"),
};

let mode = "rag";

function syncLabels() {
  els.tempVal.textContent = Number(els.temperature.value).toFixed(2);
  els.tokVal.textContent = els.maxTokens.value;
  els.kVal.textContent = els.topK.value;
  els.scoreVal.textContent = Number(els.minScore.value).toFixed(2);
}

["temperature", "maxTokens", "topK", "minScore"].forEach((id) => {
  els[id].addEventListener("input", syncLabels);
});

document.querySelectorAll(".seg").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg").forEach((b) => b.classList.remove("on"));
    btn.classList.add("on");
    mode = btn.dataset.mode;
    els.ragControls.style.display = mode === "rag" ? "block" : "none";
    els.modeHint.textContent =
      mode === "rag"
        ? "Answers are grounded in the vector store, with citations."
        : "Raw generation from the instruction model. No retrieval.";
  });
});

function addMessage(role, text, extra = "") {
  if (els.empty) els.empty.remove();
  const wrap = document.createElement("article");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `
    <div class="meta"><span>${role === "user" ? "Investigator" : "Lab model"}</span><span>${extra}</span></div>
    <div class="body"></div>
  `;
  wrap.querySelector(".body").textContent = text;
  els.thread.appendChild(wrap);
  els.thread.scrollTop = els.thread.scrollHeight;
  return wrap;
}

function addSources(wrap, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  const items = sources
    .map(
      (s, i) =>
        `<li><strong>[${i + 1}] ${s.source || "document"}</strong> · score ${Number(s.similarity_score || 0).toFixed(3)} · ${s.preview || ""}</li>`
    )
    .join("");
  box.innerHTML = `<div>Citations</div><ul>${items}</ul>`;
  wrap.appendChild(box);
}

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const live = data.status === "ok" && data.pipeline === "ready";
    els.statusPill.classList.toggle("live", live);
    els.statusPill.classList.toggle("down", !live);
    els.statusLabel.textContent = live ? "System live" : "Model loading";
    els.factModel.textContent = data.model_name || "—";
    els.factEmbed.textContent = data.embed_model || "—";
    els.factDevice.textContent = data.device || "—";
  } catch {
    els.statusPill.classList.add("down");
    els.statusLabel.textContent = "API unreachable";
  }
}

async function loadDefaults() {
  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    els.temperature.value = cfg.llm.temperature;
    els.maxTokens.value = cfg.llm.max_new_tokens;
    els.doSample.checked = cfg.llm.do_sample;
    els.topK.value = cfg.retriever.top_k;
    els.minScore.value = cfg.retriever.min_score;
    els.useHistory.checked = cfg.pipeline.use_history;
    els.summarize.checked = cfg.pipeline.summarize;
    syncLabels();
  } catch {
    syncLabels();
  }
}

async function runQuery(question) {
  const payload = {
    question,
    top_k: Number(els.topK.value),
    min_score: Number(els.minScore.value),
    summarize: els.summarize.checked,
    use_history: els.useHistory.checked,
    max_new_tokens: Number(els.maxTokens.value),
  };
  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function runGenerateStream(prompt, bodyEl) {
  const res = await fetch("/generate/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      max_new_tokens: Number(els.maxTokens.value),
      temperature: Number(els.temperature.value),
      do_sample: els.doSample.checked,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    full += decoder.decode(value, { stream: true });
    bodyEl.textContent = full;
    els.thread.scrollTop = els.thread.scrollHeight;
  }
  return full;
}

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = els.prompt.value.trim();
  if (!prompt) return;
  els.send.disabled = true;
  els.runMeta.textContent = mode === "rag" ? "Retrieving + generating…" : "Streaming tokens…";
  addMessage("user", prompt, new Date().toLocaleTimeString());
  els.prompt.value = "";
  const reply = addMessage("assistant", mode === "rag" ? "Working…" : "", "live");
  const body = reply.querySelector(".body");
  try {
    if (mode === "rag") {
      const data = await runQuery(prompt);
      body.textContent = data.answer || "";
      addSources(reply, data.sources);
      reply.querySelector(".meta span:last-child").textContent = `${data.elapsed_seconds}s`;
    } else {
      await runGenerateStream(prompt, body);
      reply.querySelector(".meta span:last-child").textContent = "complete";
    }
    els.runMeta.textContent = "Ready";
  } catch (err) {
    body.textContent = String(err);
    els.runMeta.textContent = "Failed";
  } finally {
    els.send.disabled = false;
    els.prompt.focus();
  }
});

els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

document.getElementById("clearBtn").addEventListener("click", () => {
  els.thread.innerHTML = `
    <div class="empty" id="emptyState">
      <p class="eyebrow">Session idle</p>
      <h2>Ask a research question</h2>
      <p>Use RAG to retrieve from the attention-paper index, or generate directly from the local instruction model.</p>
    </div>`;
  els.empty = document.getElementById("emptyState");
  els.runMeta.textContent = "Session cleared";
});

loadDefaults();
refreshHealth();
setInterval(refreshHealth, 8000);
