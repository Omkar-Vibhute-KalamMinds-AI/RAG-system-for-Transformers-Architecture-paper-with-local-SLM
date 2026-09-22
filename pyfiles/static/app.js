const STORE_KEY = "kalamminds.conversations";

const els = {  
  thread: document.getElementById("thread"),
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
  chatList: document.getElementById("chatList"),
  newChatBtn: document.getElementById("newChatBtn"),
  fileInput: document.getElementById("fileInput"),
  attachBtn: document.getElementById("attachBtn"),
  attachList: document.getElementById("attachList"),
  openSettingsBtn: document.getElementById("openSettingsBtn"),
  closeSettingsBtn: document.getElementById("closeSettingsBtn"),
  settingsDrawer: document.getElementById("settingsDrawer"),
  settingsBackdrop: document.getElementById("settingsBackdrop"),
};

const FILE_ACCEPT = {
  image: ["image/jpeg", "image/png", "image/gif", "image/webp"],
  document: [
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ],
};
const FILE_EXT = {
  image: ["jpg", "jpeg", "png", "gif", "webp"],
  document: ["pdf", "txt", "md", "doc", "docx"],
};
const MAX_ATTACHMENTS = 12;
const MAX_FILE_BYTES = 25 * 1024 * 1024;

/** @type {{ id: string, file: File, kind: 'image'|'document', previewUrl: string|null }[]} */
let pendingAttachments = [];
                   
let mode = "rag";
let store = loadStore();
let busy = false;

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function fileKind(file) {
  if (FILE_ACCEPT.image.includes(file.type)) return "image";
  if (FILE_ACCEPT.document.includes(file.type)) return "document";
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (FILE_EXT.image.includes(ext)) return "image";
  if (FILE_EXT.document.includes(ext)) return "document";
  return null;
}

function addFiles(fileList) {
  const rejected = [];
  for (const file of fileList) {
    if (pendingAttachments.length >= MAX_ATTACHMENTS) {
      rejected.push(`${file.name} (limit ${MAX_ATTACHMENTS})`);
      continue;
    }
    if (file.size > MAX_FILE_BYTES) {
      rejected.push(`${file.name} (too large)`);
      continue;
    }
    const kind = fileKind(file);
    if (!kind) {
      rejected.push(`${file.name} (unsupported type)`);
      continue;
    }
    if (pendingAttachments.some((a) => a.file.name === file.name && a.file.size === file.size)) continue;
    const previewUrl = kind === "image" ? URL.createObjectURL(file) : null;
    pendingAttachments.push({ id: uid(), file, kind, previewUrl });
  }
  renderAttachmentList();
  if (rejected.length) els.runMeta.textContent = rejected.slice(0, 2).join("; ");
}

function removeAttachment(id) {
  const item = pendingAttachments.find((a) => a.id === id);
  if (item?.previewUrl) URL.revokeObjectURL(item.previewUrl);
  pendingAttachments = pendingAttachments.filter((a) => a.id !== id);
  renderAttachmentList();
}

function clearPendingAttachments(revokeUrls = true) {
  if (revokeUrls) {
    pendingAttachments.forEach((a) => {
      if (a.previewUrl) URL.revokeObjectURL(a.previewUrl);
    });
  }
  pendingAttachments = [];
  renderAttachmentList();
}

function renderAttachmentList() {
  els.attachList.innerHTML = "";
  if (!pendingAttachments.length) {
    els.attachList.hidden = true;
    return;
  }
  els.attachList.hidden = false;
  pendingAttachments.forEach((item) => {
    const li = document.createElement("li");
    li.className = "attach-chip";
    if (item.kind === "image" && item.previewUrl) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.src = item.previewUrl;
      img.alt = "";
      li.appendChild(img);
    } else {
      const icon = document.createElement("span");
      icon.className = "doc-icon";
      icon.textContent = "DOC";
      li.appendChild(icon);
    }
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.file.name;
    name.title = `${item.kind} · ${formatBytes(item.file.size)}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove";
    remove.setAttribute("aria-label", `Remove ${item.file.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", (e) => {
      e.stopPropagation();
      removeAttachment(item.id);
    });
    li.append(name, remove);
    els.attachList.appendChild(li);
  });
}

function snapshotAttachments() {
  return pendingAttachments.map((a) => ({
    id: a.id,
    name: a.file.name,
    kind: a.kind,
    size: a.file.size,
    previewUrl: a.previewUrl,
  }));
}

function setupUploadUi() {
  els.attachBtn.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => {
    if (els.fileInput.files?.length) addFiles(els.fileInput.files);
    els.fileInput.value = "";
  });

  ["dragleave", "drop"].forEach((ev) => {
    els.composer.addEventListener(ev, (e) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      e.stopPropagation();
      if (ev === "drop" && e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
      els.composer.classList.remove("drag-over");
      if (ev === "drop") els.runMeta.textContent = "Ready";
    });
  });

  ["dragenter", "dragover"].forEach((ev) => {
    els.composer.addEventListener(ev, (e) => {
      if (!e.dataTransfer?.types?.includes("Files")) return;
      e.preventDefault();
      e.stopPropagation();
      els.composer.classList.add("drag-over");
      els.runMeta.textContent = "Release to attach";
    });
  });
}

function openSettings() {
  els.settingsDrawer.classList.add("open");
  els.settingsBackdrop.hidden = false;
  els.settingsDrawer.setAttribute("aria-hidden", "false");
  els.openSettingsBtn.setAttribute("aria-expanded", "true");
}

function closeSettings() {
  els.settingsDrawer.classList.remove("open");
  els.settingsBackdrop.hidden = true;
  els.settingsDrawer.setAttribute("aria-hidden", "true");
  els.openSettingsBtn.setAttribute("aria-expanded", "false");
}

function setupSettingsDrawer() {
  els.openSettingsBtn.addEventListener("click", openSettings);
  els.closeSettingsBtn.addEventListener("click", closeSettings);
  els.settingsBackdrop.addEventListener("click", closeSettings);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && els.settingsDrawer.classList.contains("open")) closeSettings();
  });
}

function uid() {
  return crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
}

function loadStore() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY));
    if (raw && Array.isArray(raw.conversations)) return raw;
  } catch (_) {}
  return { conversations: [], activeId: null };
}

function persist() {
  localStorage.setItem(STORE_KEY, JSON.stringify(store));
}

function activeChat() {
  return store.conversations.find((c) => c.id === store.activeId) || null;
}

function titleFrom(text) {
  const t = (text || "New chat").replace(/\s+/g, " ").trim();
  return t.length > 42 ? t.slice(0, 42) + "…" : t;
}

function ensureChat() {
  let chat = activeChat();
  if (chat) return chat;
  chat = { id: uid(), title: "New chat", updatedAt: Date.now(), mode, messages: [] };
  store.conversations.unshift(chat);
  store.activeId = chat.id;
  persist();
  return chat;
}

function setMode(next, fromUi) {
  mode = next;
  const chat = activeChat();
  if (chat) {
    chat.mode = next;
    persist();
  }
  document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("on", b.dataset.mode === next));
  els.ragControls.style.display = next === "rag" ? "block" : "none";
  els.modeHint.textContent =
    next === "rag"
      ? "Answers are grounded in the vector store, with citations."
      : "Raw generation from the instruction model. No retrieval.";
  if (fromUi === false) return;
}

function syncLabels() {
  els.tempVal.textContent = Number(els.temperature.value).toFixed(2);
  els.tokVal.textContent = els.maxTokens.value;
  els.kVal.textContent = els.topK.value;
  els.scoreVal.textContent = Number(els.minScore.value).toFixed(2);
}

function emptyHtml() {
  return `<div class="empty" id="emptyState">
    <p class="eyebrow">Session idle</p>
    <h2>Ask a research question</h2>
    <p>Use RAG to retrieve from the attention-paper index, or generate directly from the local instruction model.</p>
  </div>`;
}

function displayedUserText(msg) {
  if (msg.role !== "user") return msg.text;
  const versions = msg.versions && msg.versions.length ? msg.versions : [msg.text];
  const i = Math.min(msg.cursor ?? versions.length - 1, versions.length - 1);
  return versions[i];
}

function renderSidebar() {
  els.chatList.innerHTML = "";
  store.conversations.forEach((c) => {
    const row = document.createElement("div");
    row.className = "chat-item" + (c.id === store.activeId ? " active" : "");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "title";
    open.textContent = c.title || "New chat";
    open.addEventListener("click", () => switchChat(c.id));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "del";
    del.setAttribute("aria-label", "Delete chat");
    del.textContent = "×";
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(c.id);
    });
    row.append(open, del);
    els.chatList.appendChild(row);
  });
}

function renderThread() {
  const chat = activeChat();
  if (!chat || !chat.messages.length) {
    els.thread.innerHTML = emptyHtml();
    return;
  }
  els.thread.innerHTML = "";
  chat.messages.forEach((msg, idx) => {
    const wrap = document.createElement("article");
    wrap.className = `msg ${msg.role}`;
    wrap.dataset.id = msg.id;
    const meta = document.createElement("div");
    meta.className = "meta";
    const who = document.createElement("span");
    who.textContent = msg.role === "user" ? "Investigator" : "Lab model";
    const actions = document.createElement("div");
    actions.className = "actions";
    if (msg.role === "user") {
      const versions = msg.versions && msg.versions.length ? msg.versions : [msg.text];
      const cursor = msg.cursor ?? versions.length - 1;
      if (versions.length > 1) {
        const prev = document.createElement("button");
        prev.type = "button";
        prev.className = "mini";
        prev.textContent = "Prev";
        prev.disabled = cursor <= 0;
        prev.addEventListener("click", () => cycleVersion(idx, -1));
        const next = document.createElement("button");
        next.type = "button";
        next.className = "mini";
        next.textContent = "Next";
        next.disabled = cursor >= versions.length - 1;
        next.addEventListener("click", () => cycleVersion(idx, 1));
        const ver = document.createElement("span");
        ver.className = "ver";
        ver.textContent = `${cursor + 1}/${versions.length}`;
        actions.append(prev, ver, next);
        const regen = document.createElement("button");
        regen.type = "button";
        regen.className = "mini";
        regen.textContent = "Regenerate";
        regen.addEventListener("click", () => regenerateFrom(idx));
        actions.append(regen);
      }
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "mini";
      edit.textContent = "Edit";
      edit.addEventListener("click", () => startEdit(idx, wrap));
      actions.append(edit);
    } else {
      const stamp = document.createElement("span");
      stamp.textContent = msg.extra || "";
      actions.append(stamp);
    }
    meta.append(who, actions);
    if (msg.role === "user" && msg.attachments && msg.attachments.length) {
      const preview = document.createElement("div");
      preview.className = "attach-preview";
      msg.attachments.forEach((a) => {
        if (a.kind === "image" && a.previewUrl) {
          const img = document.createElement("img");
          img.src = a.previewUrl;
          img.alt = a.name;
          preview.appendChild(img);
        } else {
          const tag = document.createElement("span");
          tag.className = "doc-tag";
          tag.textContent = a.name;
          preview.appendChild(tag);
        }
      });
      wrap.appendChild(preview);
    }
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = displayedUserText(msg);
    wrap.append(meta, body);
    if (msg.role === "assistant" && msg.sources && msg.sources.length) {
      const box = document.createElement("div");
      box.className = "sources";
      const items = msg.sources
        .map(
          (s, i) =>
            `<li><strong>[${i + 1}] ${s.source || "document"}</strong> · rank ${s.rank ?? i + 1} · ${s.preview || ""}</li>`
        )
        .join("");
      box.innerHTML = `<div>Citations</div><ul>${items}</ul>`;
      wrap.appendChild(box);
    }
    els.thread.appendChild(wrap);
  });
  els.thread.scrollTop = els.thread.scrollHeight;
}

function startEdit(idx, wrap) {
  const chat = activeChat();
  const msg = chat.messages[idx];
  const current = displayedUserText(msg);
  const body = wrap.querySelector(".body");
  const box = document.createElement("textarea");
  box.className = "edit-box";
  box.value = current;
  const save = document.createElement("button");
  save.type = "button";
  save.className = "mini";
  save.textContent = "Save";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "mini";
  cancel.textContent = "Cancel";
  body.replaceWith(box);
  const actions = wrap.querySelector(".actions");
  actions.innerHTML = "";
  actions.append(save, cancel);
  box.focus();
  cancel.addEventListener("click", () => renderThread());
  save.addEventListener("click", async () => {
    const next = box.value.trim();
    if (!next || next === current) {
      renderThread();
      return;
    }
    const versions = msg.versions && msg.versions.length ? msg.versions.slice() : [msg.text];
    versions.push(next);
    msg.versions = versions;
    msg.cursor = versions.length - 1;
    msg.text = next;
    chat.messages = chat.messages.slice(0, idx + 1);
    chat.updatedAt = Date.now();
    persist();
    renderThread();
    await regenerateFrom(idx);
  });
}

function cycleVersion(idx, delta) {
  const chat = activeChat();
  const msg = chat.messages[idx];
  const versions = msg.versions && msg.versions.length ? msg.versions : [msg.text];
  const cursor = msg.cursor ?? versions.length - 1;
  const next = cursor + delta;
  if (next < 0 || next >= versions.length) return;
  msg.cursor = next;
  msg.text = versions[next];
  persist();
  renderThread();
}

async function regenerateFrom(idx) {
  const chat = activeChat();
  const msg = chat.messages[idx];
  const prompt = displayedUserText(msg);
  chat.messages = chat.messages.slice(0, idx + 1);
  persist();
  renderThread();
  await runAssistant(prompt);
}

function switchChat(id) {
  store.activeId = id;
  const chat = activeChat();
  if (chat && chat.mode) setMode(chat.mode, false);
  persist();
  renderSidebar();
  renderThread();
}

function deleteChat(id) {
  store.conversations = store.conversations.filter((c) => c.id !== id);
  if (store.activeId === id) {
    store.activeId = store.conversations[0] ? store.conversations[0].id : null;
  }
  persist();
  if (!store.activeId) newChat();
  else {
    renderSidebar();
    renderThread();
  }
}

function newChat() {
  const chat = { id: uid(), title: "New chat", updatedAt: Date.now(), mode, messages: [] };
  store.conversations.unshift(chat);
  store.activeId = chat.id;
  persist();
  renderSidebar();
  renderThread();
  els.prompt.focus();
}

["temperature", "maxTokens", "topK", "minScore"].forEach((id) => {
  els[id].addEventListener("input", syncLabels);
});

document.querySelectorAll(".seg").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

els.newChatBtn.addEventListener("click", newChat);

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

async function readPlainStream(res, bodyEl) {
  if (!res.ok) throw new Error(await res.text());
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    full += decoder.decode(value, { stream: true });
    if (bodyEl && full) bodyEl.textContent = full;
    els.thread.scrollTop = els.thread.scrollHeight;
  }
  return full;
}

async function runQueryStream(question, bodyEl) {
  const res = await fetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: Number(els.topK.value),
      min_score: Number(els.minScore.value),
      summarize: els.summarize.checked,
      use_history: els.useHistory.checked,
      max_new_tokens: Number(els.maxTokens.value),
      stream: true,
    }),
  });
  return readPlainStream(res, bodyEl);
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
  return readPlainStream(res, bodyEl);
}

async function runAssistant(prompt) {
  const chat = ensureChat();
  busy = true;
  els.send.disabled = true;
  els.runMeta.textContent = "Working…";
  const assistant = {
    id: uid(),
    role: "assistant",
    text: "Working…",
    sources: [],
    extra: "live",
  };
  chat.messages.push(assistant);
  persist();
  renderThread();
  const body = els.thread.querySelector(`.msg[data-id="${assistant.id}"] .body`);
  try {
    if (mode === "rag") {
      assistant.text = await runQueryStream(prompt, body);
      assistant.extra = "complete";
    } else {
      assistant.text = await runGenerateStream(prompt, body);
      assistant.extra = "complete";
    }
    chat.updatedAt = Date.now();
    persist();
    renderThread();
    els.runMeta.textContent = "Ready";
  } catch (err) {
    assistant.text = String(err);
    assistant.extra = "failed";
    persist();
    renderThread();
    els.runMeta.textContent = "Failed";
  } finally {
    busy = false;
    els.send.disabled = false;
    els.prompt.focus();
  }
}

els.composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (busy) return;
  const prompt = els.prompt.value.trim();
  const attachments = snapshotAttachments();
  if (!prompt && !attachments.length) return;
  const chat = ensureChat();
  chat.mode = mode;
  const displayText = prompt || "(Attached files only)";
  if (chat.title === "New chat") chat.title = titleFrom(displayText);
  chat.messages.push({
    id: uid(),
    role: "user",
    text: displayText,
    versions: [displayText],
    cursor: 0,
    attachments,
  });
  chat.updatedAt = Date.now();
  const queryText =
    prompt +
    (attachments.length
      ? `\n\n[Attached: ${attachments.map((a) => a.name).join(", ")}]`
      : "");
  persist();
  els.prompt.value = "";
  clearPendingAttachments(false);
  renderSidebar();
  renderThread();
  await runAssistant(queryText.trim() || displayText);
});

setupUploadUi();
setupSettingsDrawer();

els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

if (!store.activeId && store.conversations.length) {
  store.activeId = store.conversations[0].id;
}
if (!store.conversations.length) newChat();
else {
  const chat = activeChat();
  if (chat && chat.mode) setMode(chat.mode, false);
  renderSidebar();
  renderThread();
}

loadDefaults();
refreshHealth();
setInterval(refreshHealth, 8000);
