import { chunkText, buildIndex, search, stripGutenberg } from "./search.js?v=3";

const MODELS = [
  { id: "Llama-3.2-1B-Instruct-q4f16_1-MLC", f32: "Llama-3.2-1B-Instruct-q4f32_1-MLC", name: "Llama 3.2 1B (recommended)", size: "about 0.7 GB download" },
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", f32: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC", name: "Qwen 2.5 0.5B (smallest, fastest)", size: "about 0.3 GB download" },
  { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", f32: "Llama-3.2-3B-Instruct-q4f32_1-MLC", name: "Llama 3.2 3B (smarter, slower)", size: "about 1.8 GB download" },
];
const TOP_K = 3;
const HISTORY_TURNS = 2;   // earlier question/answer pairs sent along, to allow follow-up questions
const SNIPPET_PREVIEW = 400;

const $ = (id) => document.getElementById(id);
const el = {
  modelSelect: $("model-select"), modelHelp: $("model-help"), loadBtn: $("load-btn"),
  modelProgress: $("model-progress"), modelFill: $("model-progress-fill"), modelLabel: $("model-progress-label"),
  gpuNotice: $("gpu-notice"), modelStatus: $("model-status"),
  bookInput: $("book-input"), bookDrop: $("book-drop"),
  bookProgress: $("book-progress"), bookFill: $("book-progress-fill"), bookLabel: $("book-progress-label"),
  bookStatus: $("book-status"),
  log: $("rag-log"), empty: $("rag-empty"), form: $("rag-form"), input: $("rag-input"),
  send: $("rag-send"), stop: $("rag-stop"), clear: $("rag-clear"),
};

const state = {
  worker: null,
  modelReady: false,
  modelLoading: false,
  hasF16: true,
  index: null,
  bookName: "",
  generating: null,   // {id, bubble, text}
  history: [],        // [{role, content}] without the system prompt
  nextId: 1,
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function setProgress(fill, label, frac, text) {
  fill.style.width = Math.round(Math.min(1, Math.max(0, frac)) * 100) + "%";
  label.textContent = text;
}
function show(node, on) { node.classList.toggle("hidden", !on); }

// ---------------------------------------------------------------------------
// Step 1: the model (runs in a Web Worker)
// ---------------------------------------------------------------------------
for (const m of MODELS) {
  const o = document.createElement("option");
  o.value = m.id;
  o.textContent = `${m.name}, ${m.size}`;
  el.modelSelect.appendChild(o);
}

function modelId() {
  const m = MODELS.find(x => x.id === el.modelSelect.value) || MODELS[0];
  return state.hasF16 ? m.id : m.f32;
}

async function checkWebGPU() {
  if (!("gpu" in navigator)) {
    el.gpuNotice.innerHTML = `<strong>Your browser does not support WebGPU</strong>, so the AI cannot run here. Try a recent Chrome or Edge on a computer. You can still upload a book and search it: questions will show the best matching passages.`;
    show(el.gpuNotice, true);
    el.loadBtn.disabled = true;
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no adapter");
    state.hasF16 = adapter.features.has("shader-f16");
  } catch {
    el.gpuNotice.innerHTML = `<strong>WebGPU is turned off or not available on this device.</strong> The AI cannot run, but you can still search a book.`;
    show(el.gpuNotice, true);
    el.loadBtn.disabled = true;
    return false;
  }
  return true;
}

function makeWorker() {
  const w = new Worker(new URL("./rag-worker.js?v=3", import.meta.url), { type: "module" });
  w.onmessage = onWorkerMessage;
  w.onerror = (e) => {
    e.preventDefault();
    modelFailed(e.message || "The AI worker could not start. Check your internet connection and reload the page.");
  };
  return w;
}

function onWorkerMessage(e) {
  const m = e.data;
  switch (m.type) {
    case "progress":
      setProgress(el.modelFill, el.modelLabel, m.progress, m.text);
      break;
    case "ready":
      state.modelReady = true;
      state.modelLoading = false;
      setProgress(el.modelFill, el.modelLabel, 1, "Ready.");
      setTimeout(() => show(el.modelProgress, false), 800);
      el.modelStatus.innerHTML = `<span class="ok">✓</span> <strong>${escapeHtml(m.model)}</strong> is running on your device.`;
      el.loadBtn.textContent = "Switch model";
      el.loadBtn.disabled = false;
      el.modelSelect.disabled = false;
      updateChatEnabled();
      break;
    case "delta":
      if (state.generating && state.generating.id === m.id) appendDelta(m.text);
      break;
    case "done":
      if (state.generating && state.generating.id === m.id) finishAnswer(m.usage);
      break;
    case "error":
      if (m.during === "load") modelFailed(m.message);
      else if (state.generating && (m.id === undefined || state.generating.id === m.id)) failAnswer(m.message);
      break;
  }
}

function modelFailed(message) {
  state.modelLoading = false;
  state.modelReady = false;
  show(el.modelProgress, false);
  el.modelStatus.innerHTML = `<span class="bad">✗</span> Could not load the model: ${escapeHtml(message)}`;
  el.loadBtn.disabled = false;
  el.modelSelect.disabled = false;
  if (state.generating) failAnswer(message);
  updateChatEnabled();
}

el.loadBtn.addEventListener("click", () => {
  if (state.modelLoading || state.generating) return;
  if (!state.worker) state.worker = makeWorker();
  state.modelLoading = true;
  state.modelReady = false;
  el.loadBtn.disabled = true;
  el.modelSelect.disabled = true;
  el.modelStatus.textContent = "";
  show(el.modelProgress, true);
  setProgress(el.modelFill, el.modelLabel, 0, "Starting… (the first time, this downloads the model)");
  updateChatEnabled();
  state.worker.postMessage({ type: "load", model: modelId() });
});

el.modelSelect.addEventListener("change", () => {
  const m = MODELS.find(x => x.id === el.modelSelect.value);
  el.modelHelp.textContent = `${m.size[0].toUpperCase() + m.size.slice(1)}. It downloads once and is saved in your browser for next time.`;
});
el.modelSelect.dispatchEvent(new Event("change"));

// ---------------------------------------------------------------------------
// Step 2: the book
// ---------------------------------------------------------------------------
function readFileAsText(file, onProgress) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read " + file.name));
    reader.readAsText(file);
  });
}

async function loadBook(fileList) {
  const files = Array.from(fileList).filter(f => /\.txt$/i.test(f.name) || f.type === "text/plain");
  if (!files.length) {
    el.bookStatus.innerHTML = `<span class="bad">✗</span> Please choose a .txt file.`;
    return;
  }
  show(el.bookProgress, true);
  el.bookStatus.textContent = "";
  state.index = null;
  updateChatEnabled();
  try {
    const texts = [];
    for (let i = 0; i < files.length; i++) {
      const t = await readFileAsText(files[i], (f) =>
        setProgress(el.bookFill, el.bookLabel, 0.4 * (i + f) / files.length, `Reading ${files[i].name}…`));
      texts.push(stripGutenberg(t));
    }
    setProgress(el.bookFill, el.bookLabel, 0.4, "Cutting the text into chunks…");
    await new Promise(r => setTimeout(r, 0));
    const chunks = chunkText(texts.join("\n\n"));
    if (!chunks.length) throw new Error("No text found in that file.");
    const index = await buildIndex(chunks, (f) =>
      setProgress(el.bookFill, el.bookLabel, 0.4 + 0.6 * f, `Building the search index… ${Math.round(f * 100)}%`));
    state.index = index;
    state.bookName = files.map(f => f.name).join(", ");
    const words = chunks.reduce((a, c) => a + c.split(/\s+/).length, 0);
    el.bookStatus.innerHTML = `<span class="ok">✓</span> <strong>${escapeHtml(state.bookName)}</strong>: ${words.toLocaleString()} words in ${chunks.length.toLocaleString()} chunks.`;
    setTimeout(() => show(el.bookProgress, false), 500);
  } catch (err) {
    show(el.bookProgress, false);
    el.bookStatus.innerHTML = `<span class="bad">✗</span> ${escapeHtml(err.message || String(err))}`;
  }
  updateChatEnabled();
}

el.bookInput.addEventListener("change", () => loadBook(el.bookInput.files));
["dragenter", "dragover"].forEach(ev => el.bookDrop.addEventListener(ev, e => { e.preventDefault(); el.bookDrop.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => el.bookDrop.addEventListener(ev, e => { e.preventDefault(); el.bookDrop.classList.remove("drag"); }));
el.bookDrop.addEventListener("drop", e => loadBook(e.dataTransfer.files));

// ---------------------------------------------------------------------------
// Step 3: chat
// ---------------------------------------------------------------------------
function updateChatEnabled() {
  const canAsk = !!state.index && !state.generating;
  el.input.disabled = !state.index;
  el.send.disabled = !canAsk;
  if (!state.index) el.input.placeholder = "Upload a book first…";
  else if (!state.modelReady) el.input.placeholder = "Ask a question (the AI isn't loaded yet, so it will only search)…";
  else el.input.placeholder = "Ask a question about the book…";
}

function buildSystemPrompt(results) {
  const snippets = results.map((r, i) => `[Snippet ${i + 1}]\n${r.text}`).join("\n\n");
  return "You are a helpful assistant. Answer the user's question ONLY using the following text snippets. " +
    "If the answer is not in the text, say you do not know.\n\n" + snippets;
}

function addMessage(cls, html) {
  show(el.empty, false);
  const d = document.createElement("div");
  d.className = "msg " + cls;
  if (html != null) d.innerHTML = html;
  el.log.appendChild(d);
  el.log.scrollTop = el.log.scrollHeight;
  return d;
}

function highlight(text, matched) {
  let html = escapeHtml(text.length > SNIPPET_PREVIEW ? text.slice(0, SNIPPET_PREVIEW) + "…" : text);
  if (!matched.length) return html;
  // Highlight words that start with a matched search term.
  const re = new RegExp(`(?<![&#\\p{L}\\p{N}])(${matched.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})[\\p{L}]*`, "giu");
  return html.replace(re, "<mark>$&</mark>");
}

function sourcesHtml(results, systemPrompt) {
  if (!results.length) return "";
  const n = state.index.chunks.length;
  return `<details class="sources">
      <summary>Sources: ${results.length} passage${results.length === 1 ? "" : "s"} found</summary>
      ${results.map((r, i) => `
        <div class="source">
          <div class="source-head">#${i + 1} · chunk ${(r.id + 1).toLocaleString()} of ${n.toLocaleString()} (${Math.round(100 * r.id / n)}% into the book) · score ${r.score.toFixed(2)}</div>
          <div class="source-text">${highlight(r.text, r.matched)}</div>
        </div>`).join("")}
      ${systemPrompt ? `<details class="prompt"><summary>Show the hidden prompt</summary><pre>${escapeHtml(systemPrompt)}</pre></details>` : ""}
    </details>`;
}

el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = el.input.value.trim();
  if (!question || !state.index || state.generating) return;
  el.input.value = "";
  addMessage("user").textContent = question;

  let results = search(state.index, question, TOP_K);
  // A follow-up like "why did he do that?" may have no useful words: use the last question too.
  if (results.length < TOP_K) {
    const lastQ = [...state.history].reverse().find(m => m.role === "user");
    if (lastQ) {
      const more = search(state.index, lastQ.content + " " + question, TOP_K);
      for (const r of more) if (results.length < TOP_K && !results.some(x => x.id === r.id)) results.push(r);
    }
  }

  if (!state.modelReady) {
    const bubble = addMessage("bot");
    bubble.innerHTML = results.length
      ? `<p class="muted">The AI isn't loaded, so here are the best matching passages. Load the AI in step 1 to get real answers.</p>${sourcesHtml(results, "")}`
      : `<p class="muted">No passage matches those words. Try other words from the book.</p>`;
    const d = bubble.querySelector("details"); if (d) d.open = true;
    el.log.scrollTop = el.log.scrollHeight;
    return;
  }

  const systemPrompt = buildSystemPrompt(results);
  const messages = [
    { role: "system", content: systemPrompt },
    ...state.history.slice(-HISTORY_TURNS * 2),
    { role: "user", content: question },
  ];

  const bubble = addMessage("bot", `<div class="answer"><span class="cursor"></span></div>${sourcesHtml(results, systemPrompt)}`);
  const id = state.nextId++;
  state.generating = { id, bubble, text: "", question, answerEl: bubble.querySelector(".answer") };
  show(el.stop, true);
  updateChatEnabled();
  state.worker.postMessage({ type: "chat", id, messages, options: { temperature: 0.3, max_tokens: 512 } });
});

function renderAnswer(text, done) {
  const g = state.generating;
  g.answerEl.textContent = text;
  if (!done) g.answerEl.insertAdjacentHTML("beforeend", `<span class="cursor"></span>`);
}

function appendDelta(t) {
  const g = state.generating;
  const nearBottom = el.log.scrollHeight - el.log.scrollTop - el.log.clientHeight < 60;
  g.text += t;
  renderAnswer(g.text, false);
  if (nearBottom) el.log.scrollTop = el.log.scrollHeight;
}

function finishAnswer(usage) {
  const g = state.generating;
  renderAnswer(g.text || "(no answer)", true);
  if (usage) {
    const info = document.createElement("span");
    info.className = "how";
    const speed = usage.extra?.decode_tokens_per_s;
    info.textContent = `${usage.prompt_tokens.toLocaleString()} tokens in, ${usage.completion_tokens.toLocaleString()} out` +
      (speed ? ` · ${speed.toFixed(1)} tokens/s` : "");
    g.answerEl.after(info);
  }
  state.history.push({ role: "user", content: g.question }, { role: "assistant", content: g.text });
  endGeneration();
}

function failAnswer(message) {
  const g = state.generating;
  g.answerEl.innerHTML = `<span class="bad">Error:</span> ${escapeHtml(message)}`;
  endGeneration();
}

function endGeneration() {
  state.generating = null;
  show(el.stop, false);
  updateChatEnabled();
  el.input.focus({ preventScroll: true });
}

el.stop.addEventListener("click", () => {
  if (state.generating) state.worker.postMessage({ type: "stop" });
});

el.clear.addEventListener("click", () => {
  if (state.generating) return;
  el.log.querySelectorAll(".msg").forEach(m => m.remove());
  show(el.empty, true);
  state.history = [];
});

checkWebGPU();
updateChatEnabled();
