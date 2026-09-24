(() => {
  "use strict";

  const MAX_WORDS = 50_000_000;
  // Memory slider positions. Infinity means "as long as has ever been seen".
  const ORDERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 128, 256, 1024, Infinity];
  const BYTES_PER_WORD = 5.9; // rough average for English plain text

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const el = {
    tabs: document.querySelectorAll(".tab"),
    panels: document.querySelectorAll(".tab-panel"),
    text: $("text-input"),
    file: $("file-input"),
    drop: $("dropzone"),
    fileList: $("file-list"),
    samples: $("samples"),
    counter: $("word-counter"),
    keepLines: $("keep-lines"),
    trainBtn: $("train-btn"),
    progress: $("progress"),
    progressFill: $("progress-fill"),
    progressLabel: $("progress-label"),
    stats: $("stats"),
    chatCard: $("chat-card"),
    chatLog: $("chat-log"),
    emptyChat: $("empty-chat"),
    chatForm: $("chat-form"),
    chatInput: $("chat-input"),
    sendBtn: $("send-btn"),
    clearChat: $("clear-chat"),
    order: $("order"), orderVal: $("order-val"),
    temp: $("temp"), tempVal: $("temp-val"),
    length: $("length"), lengthVal: $("length-val"),
    avoidCopy: $("avoid-copy"),
    animate: $("animate"),
    colorToks: $("color-toks"),
    inspector: $("inspector-body"),
    exploreInput: $("explore-input"),
    exploreOut: $("explore-out"),
  };

  const state = {
    source: "paste",     // which tab's text to use
    files: [],
    sampleId: null,
    worker: null,
    ready: false,
    busy: false,
    reqId: 0,
    pending: new Map(),
    stats: null,
  };

  // ----- worker plumbing -----
  function makeWorker() {
    if (state.worker) state.worker.terminate();
    const src = `(${workerMain.toString()})();`;
    const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    const w = new Worker(url);
    URL.revokeObjectURL(url);
    w.onmessage = onWorkerMessage;
    w.onerror = (e) => {
      showProgress(false);
      alert("Something went wrong while building the model: " + (e.message || "unknown error") +
        "\n\nIf the text is very large, your browser may have run out of memory. Try a smaller text, or close other tabs.");
      state.busy = false;
      updateTrainButton();
    };
    state.worker = w;
    state.pending.clear();
  }

  function ask(msg) {
    return new Promise((resolve, reject) => {
      const id = ++state.reqId;
      state.pending.set(id, { resolve, reject });
      state.worker.postMessage({ ...msg, id });
    });
  }

  function onWorkerMessage(e) {
    const m = e.data;
    if (m.type === "progress") {
      setProgress(m.pct, m.phase, m.indeterminate);
    } else if (m.type === "built") {
      onBuilt(m.stats);
    } else if (m.id && state.pending.has(m.id)) {
      const p = state.pending.get(m.id);
      state.pending.delete(m.id);
      if (m.type === "error") p.reject(new Error(m.message));
      else p.resolve(m);
    } else if (m.type === "error") {
      showProgress(false);
      if (!state.ready) setChatEnabled(false);
      alert("Error: " + m.message);
      state.busy = false;
      updateTrainButton();
    }
  }

  // ----- tabs & input -----
  el.tabs.forEach(tab => tab.addEventListener("click", () => {
    el.tabs.forEach(t => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
    el.panels.forEach(p => p.classList.toggle("hidden", p.dataset.panel !== tab.dataset.tab));
    state.source = tab.dataset.tab;
    updateCounter();
  }));

  function countWords(text) {
    let n = 0;
    const re = /\S+/g;
    while (re.exec(text)) n++;
    return n;
  }

  function currentText() {
    if (state.source === "paste") return el.text.value;
    const s = (window.SAMPLES || []).find(x => x.id === state.sampleId);
    return s ? s.text : "";
  }

  let counterTimer = 0;
  function updateCounter() {
    clearTimeout(counterTimer);
    counterTimer = setTimeout(() => {
      let n, about = "";
      if (state.source === "upload") {
        const bytes = state.files.reduce((a, f) => a + f.size, 0);
        n = Math.round(bytes / BYTES_PER_WORD);
        about = n ? "about " : "";
      } else {
        n = countWords(currentText());
      }
      el.counter.textContent = `${about}${n.toLocaleString()} / ${MAX_WORDS.toLocaleString()} words` +
        (n > MAX_WORDS ? ` (only the first ${MAX_WORDS.toLocaleString()} will be used)` : "");
      el.counter.classList.toggle("over", n > MAX_WORDS);
      state.wordCount = n;
      updateTrainButton();
    }, 150);
  }

  function updateTrainButton() {
    el.trainBtn.disabled = state.busy || !(state.wordCount > 0);
  }

  el.text.addEventListener("input", updateCounter);

  // Files are not read here. The worker streams them, so even huge files
  // never sit in this page's memory as one giant string.
  function loadFiles(files) {
    const list = Array.from(files);
    if (!list.length) return;
    state.files = list;
    el.fileList.innerHTML = "";
    for (const f of list) {
      const li = document.createElement("li");
      li.textContent = `${f.name} (${formatBytes(f.size)})`;
      el.fileList.appendChild(li);
    }
    updateCounter();
  }

  function formatBytes(b) {
    if (b < 1024) return b + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / 1024 / 1024).toFixed(1) + " MB";
  }

  el.file.addEventListener("change", () => loadFiles(el.file.files));
  ["dragenter", "dragover"].forEach(ev => el.drop.addEventListener(ev, e => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => el.drop.addEventListener(ev, e => { e.preventDefault(); el.drop.classList.remove("drag"); }));
  el.drop.addEventListener("drop", e => loadFiles(e.dataTransfer.files));

  // Samples
  (window.SAMPLES || []).forEach(s => {
    const b = document.createElement("button");
    b.className = "sample";
    b.type = "button";
    b.innerHTML = `<strong></strong><span></span>`;
    b.querySelector("strong").textContent = s.title;
    b.querySelector("span").textContent = `${s.blurb} (${countWords(s.text).toLocaleString()} words)`;
    b.addEventListener("click", () => {
      state.sampleId = s.id;
      el.samples.querySelectorAll(".sample").forEach(x => x.classList.toggle("active", x === b));
      updateCounter();
    });
    el.samples.appendChild(b);
  });

  // ----- training -----
  function showProgress(on) { el.progress.classList.toggle("hidden", !on); }
  function setProgress(pct, label, indeterminate) {
    el.progressFill.classList.toggle("indeterminate", !!indeterminate);
    el.progressFill.style.width = Math.min(100, pct) + "%";
    el.progressLabel.textContent = label;
  }

  el.trainBtn.addEventListener("click", () => {
    const files = state.source === "upload" ? state.files : null;
    const text = files ? "" : currentText();
    if (!files && !text.trim()) return;
    state.busy = true;
    state.ready = false;
    updateTrainButton();
    setChatEnabled(false);
    el.stats.classList.add("hidden");
    showProgress(true);
    setProgress(2, "Starting…");
    makeWorker();
    state.worker.postMessage({ type: "build", text, files, maxWords: MAX_WORDS, keepLines: el.keepLines.checked });
  });

  function onBuilt(stats) {
    state.busy = false;
    state.ready = true;
    state.stats = stats;
    setProgress(100, `Done in ${(stats.ms / 1000).toFixed(1)} seconds.`);
    setTimeout(() => showProgress(false), 700);
    updateTrainButton();
    el.trainBtn.textContent = "Rebuild model";
    renderStats(stats);
    setChatEnabled(true);
    clearChat();
    el.chatInput.focus({ preventScroll: true });
    el.chatCard.scrollIntoView({ behavior: "smooth", block: "start" });
    runExplore();
  }

  function renderStats(s) {
    const small = s.words < 5000;
    el.stats.innerHTML = `
      <div class="stat-row">
        <div class="stat"><b>${s.tokens.toLocaleString()}</b><span>tokens read</span></div>
        <div class="stat"><b>${s.vocab.toLocaleString()}</b><span>different tokens (vocabulary)</span></div>
        <div class="stat"><b>${s.pairs.toLocaleString()}</b><span>different two-word patterns</span></div>
        <div class="stat"><b>${s.depth ? s.depth.toLocaleString() : "∞"}</b><span>${s.depth ? "longest memory it can use (words)" : "longest memory: unlimited"}</span></div>
        <div class="stat"><b>${(s.ms / 1000).toFixed(1)}s</b><span>to build</span></div>
      </div>
      <div class="top-words">Most common words: ${s.topWords.map(([w, c]) => `<code title="${c.toLocaleString()} times">${escapeHtml(w)}</code>`).join(" ")}</div>
      ${s.truncated ? `<p class="note"><strong>Your text was longer than ${MAX_WORDS.toLocaleString()} words, so only the first ${s.words.toLocaleString()} were used.</strong></p>` : ""}
      <p class="note muted">${small
        ? "This is a small text, so the model will often copy whole sentences from it. Try a memory of 1 or 2 words for more mixing, or give it a bigger text."
        : "For comparison: a real LLM is trained on trillions of tokens and has billions of settings. Yours read " + s.tokens.toLocaleString() + " tokens and just counts them."}</p>`;
    el.stats.classList.remove("hidden");
  }

  function setChatEnabled(on) {
    el.chatCard.classList.toggle("disabled", !on);
    el.chatInput.disabled = !on;
    el.sendBtn.disabled = !on;
    el.exploreInput.disabled = !on;
  }

  // ----- settings -----
  const LENGTH_NAMES = ["", "Short", "Medium", "Long", "Very long"];
  function settings() {
    return {
      order: ORDERS[+el.order.value] === Infinity ? 1e9 : ORDERS[+el.order.value],
      temp: +el.temp.value / 10,
      length: +el.length.value,
      avoidCopy: el.avoidCopy.checked,
    };
  }
  function syncSettingLabels() {
    const s = settings();
    el.orderVal.textContent = s.order >= 1e9 ? "Unlimited" : s.order === 1 ? "1 word" : `${s.order.toLocaleString()} words`;
    el.tempVal.textContent = s.temp === 0 ? "0 (always top)" : s.temp.toFixed(1);
    el.lengthVal.textContent = LENGTH_NAMES[s.length];
  }
  [el.order, el.temp, el.length].forEach(x => x.addEventListener("input", () => { syncSettingLabels(); runExplore(); }));
  el.avoidCopy.addEventListener("change", runExplore);
  el.colorToks.addEventListener("change", () => el.chatLog.classList.toggle("colored", el.colorToks.checked));
  el.chatLog.classList.toggle("colored", el.colorToks.checked);
  syncSettingLabels();

  // ----- chat -----
  function clearChat() {
    el.chatLog.querySelectorAll(".msg").forEach(m => m.remove());
    el.emptyChat.classList.toggle("hidden", false);
    el.emptyChat.innerHTML = state.ready
      ? `Your model is ready. Say something to it!<br><span class="muted">Tip: click any word in a reply to see why the model picked it.</span>`
      : `Build a model first. Then say something to it.<br><span class="muted">Tip: click any word in a reply to see why the model picked it.</span>`;
    el.inspector.className = "inspector-empty";
    el.inspector.textContent = "Click a word in one of the model's replies to see the choices it had.";
  }
  el.clearChat.addEventListener("click", clearChat);

  function addMsg(cls, text) {
    el.emptyChat.classList.add("hidden");
    const d = document.createElement("div");
    d.className = "msg " + cls;
    if (text != null) d.textContent = text;
    el.chatLog.appendChild(d);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
    return d;
  }

  el.chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = el.chatInput.value.trim();
    if (!text || !state.ready || state.generating) return;
    el.chatInput.value = "";
    addMsg("user", text);
    state.generating = true;
    el.sendBtn.disabled = true;
    const bubble = addMsg("bot");
    bubble.innerHTML = `<span class="muted">…</span>`;
    try {
      const res = await ask({ type: "generate", prompt: text, settings: settings() });
      await renderReply(bubble, res);
    } catch (err) {
      bubble.classList.add("error");
      bubble.textContent = "Error: " + err.message;
    }
    state.generating = false;
    el.sendBtn.disabled = false;
    el.chatInput.focus({ preventScroll: true });
  });

  // Decide spacing between tokens so "hello , world" reads "hello, world".
  const NO_SPACE_BEFORE = /^([.,!?;:%)\]}»…]|\.{2,}|['’]\p{L}+|n['’]t)$/u;
  const NO_SPACE_AFTER = /^[(\[{«$#@]$/;
  function makeJoiner() {
    let prev = null;
    let dq = 0; // open straight double quotes
    return (tok) => {
      if (tok === "\u2029") { prev = "\u2029"; return { br: true }; }
      let space = prev !== null && prev !== "\u2029";
      let isOpen = false;
      if (tok === '"') {
        if (dq % 2 === 0) { isOpen = true; } else { space = false; }
        dq++;
      } else if (tok === "“") isOpen = true;
      else if (tok === "”") space = false;
      if (NO_SPACE_BEFORE.test(tok)) space = false;
      if (prev && (NO_SPACE_AFTER.test(prev) || prev === "“" || prev === "\u0001")) space = false;
      if (tok === "-" || tok === "—" || tok === "/") space = tok === "—" ? space : false;
      if (prev === "-" || prev === "/") space = false;
      prev = isOpen ? "\u0001" : tok;
      return { space };
    };
  }

  function probClass(p) {
    if (p >= 0.8) return "p-hi";
    if (p >= 0.25) return "p-mid";
    return "p-lo";
  }

  async function renderReply(bubble, res) {
    bubble.innerHTML = "";
    const body = document.createElement("span");
    bubble.appendChild(body);
    const join = makeJoiner();
    const animate = el.animate.checked;
    let first = true;
    const steps = res.steps;
    if (!steps.length) {
      body.textContent = "(The model had nothing to say. Try a different message.)";
      bubble.classList.add("error");
    }
    for (const step of steps) {
      const j = join(step.t);
      if (j.br) { body.appendChild(document.createElement("br")); continue; }
      if (j.space) body.appendChild(document.createTextNode(" "));
      const span = document.createElement("span");
      let t = step.t;
      if (first && res.how.mode !== "continue" && /^\p{Ll}/u.test(t)) t = t[0].toUpperCase() + t.slice(1);
      first = false;
      span.textContent = t;
      span.className = "tok " + (step.seed ? "seed" : probClass(step.info.p));
      span._step = step;
      span.addEventListener("click", () => {
        el.chatLog.querySelectorAll(".tok.selected").forEach(x => x.classList.remove("selected"));
        span.classList.add("selected");
        showStep(step);
      });
      body.appendChild(span);
      if (animate) {
        el.chatLog.scrollTop = el.chatLog.scrollHeight;
        await new Promise(r => setTimeout(r, 45));
      }
    }
    const how = document.createElement("span");
    how.className = "how";
    if (res.how.mode === "continue") how.textContent = `Continued from the last ${res.how.words === 1 ? "word" : res.how.words + " words"} of your message.`;
    else if (res.how.mode === "keyword") how.textContent = `Started from your word "${res.how.word}". The end of your message wasn't in its text, so it started from your rarest word that it knows.`;
    else how.textContent = "It didn't know any of your words, so it started a new sentence.";
    bubble.appendChild(how);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function showTok(t) { return t === "\u2029" ? "↵" : t; }

  function barsHtml(info, clickable) {
    const maxP = Math.max(...info.top.map(x => x.p), 0.0001);
    return `<div class="bars">${info.top.map(o => `
      <div class="bar-row${o.t === info.chosen ? " chosen" : ""}${clickable ? " clickable" : ""}" data-t="${escapeHtml(o.t)}" title="seen ${o.c.toLocaleString()} times">
        <span class="w">${escapeHtml(showTok(o.t))}</span>
        <span class="track"><span class="fill" style="width:${(100 * o.p / maxP).toFixed(1)}%; display:block"></span></span>
        <span class="p">${fmtP(o.p)}</span>
      </div>`).join("")}</div>`;
  }

  function fmtP(p) {
    if (p >= 0.995) return "100%";
    if (p >= 0.1) return Math.round(p * 100) + "%";
    if (p >= 0.01) return (p * 100).toFixed(1) + "%";
    return p > 0 ? "<1%" : "0%";
  }

  function contextHtml(info, fullCtx) {
    if (info.order === 0) {
      return `<div class="ctx-line">It found no match for the words before, so it used <strong>0 words</strong> of memory and picked from all words by how common they are.</div>`;
    }
    const hidden = info.order - info.ctx.length;
    return `<div class="ctx-line">It looked at the last <strong>${info.order.toLocaleString()} ${info.order === 1 ? "word" : "words"}</strong>:<br>${
      hidden > 0 ? `<span class="muted">…${hidden.toLocaleString()} more before… </span>` : ""}${
      info.ctx.map(w => `<span class="ctx-chip">${escapeHtml(showTok(w))}</span>`).join("")}</div>`;
  }

  function showStep(step) {
    el.inspector.className = "";
    if (step.seed) {
      el.inspector.innerHTML = `<div class="ctx-line"><strong>"${escapeHtml(step.t)}"</strong> was not predicted. The model did not recognise the end of your message, so it took the rarest word from your message that appears in its text (seen ${step.seedCount.toLocaleString()} times) and started from there.</div>
        <p class="small-note">Real chatbots don't need this trick. They can understand any message.</p>`;
      return;
    }
    const info = step.info;
    const s = settings();
    let backedOff = "";
    if (s.order >= 1e9) {
      backedOff = `<p class="small-note">Memory is unlimited, so it used the longest run of recent words that also appears in your text${s.avoidCopy ? " with at least two different next words" : ""}.</p>`;
    } else if (info.order < s.order) {
      backedOff = `<p class="small-note">Memory is set to ${s.order.toLocaleString()} words, but that exact run of words was never seen${s.avoidCopy ? " (or had only one option)" : ""}, so it <strong>backed off</strong> to ${info.order.toLocaleString()}.</p>`;
    }
    el.inspector.innerHTML = `
      ${contextHtml(info)}
      ${backedOff}
      <div class="ctx-line">In your text, this was followed by <strong>${info.options.toLocaleString()}</strong> different ${info.options === 1 ? "token" : "tokens"}${info.sampled ? " (counted from a sample)" : ""}. Chances after creativity is applied:</div>
      ${barsHtml(info, false)}
      <p class="small-note">It rolled the dice and got <strong>${escapeHtml(showTok(info.chosen))}</strong> (${fmtP(info.p)} chance).${info.options === 1 ? " There was only one option, so it copied your text exactly." : ""}</p>`;
  }

  // ----- explorer -----
  let exploreTimer = 0;
  function runExplore() {
    clearTimeout(exploreTimer);
    exploreTimer = setTimeout(async () => {
      if (!state.ready) return;
      const text = el.exploreInput.value;
      try {
        const res = await ask({ type: "predict", text, settings: settings() });
        const info = res.info;
        if (!info) { el.exploreOut.innerHTML = ""; return; }
        const unknown = info.unknown && info.unknown.length
          ? `<p class="small-note">Never seen: ${info.unknown.map(u => `<code>${escapeHtml(u)}</code>`).join(" ")}</p>` : "";
        const intro = text.trim()
          ? contextHtml(info)
          : `<div class="ctx-line">With no words yet, these are the most common tokens overall:</div>`;
        el.exploreOut.innerHTML = `${intro}${unknown}${barsHtml(info, true)}`;
        el.exploreOut.querySelectorAll(".bar-row").forEach(row => row.addEventListener("click", () => {
          const t = row.dataset.t;
          const v = el.exploreInput.value;
          const join = t === "\u2029" ? "\n" : (/^[.,!?;:)]/.test(t) || !v || /\s$/.test(v) ? "" : " ");
          el.exploreInput.value = v + join + (t === "\u2029" ? "" : t);
          runExplore();
        }));
      } catch (err) {
        el.exploreOut.textContent = "Error: " + err.message;
      }
    }, 120);
  }
  el.exploreInput.addEventListener("input", runExplore);

  updateCounter();
})();
