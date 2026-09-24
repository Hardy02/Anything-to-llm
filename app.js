(() => {
  "use strict";

  const MAX_WORDS = 1_000_000;
  const MAX_ORDER = 6;

  // ---------------------------------------------------------------------------
  // Model worker. The whole function is turned into a Blob so the site works
  // from GitHub Pages and from a plain file:// double-click alike.
  // ---------------------------------------------------------------------------
  function workerMain() {
    const MAX_ORDER = 6;
    const PARA = "\u2029";
    const TOKEN_RE = /\u2029|[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*|\.{2,}|[^\s\p{L}\p{N}]/gu;
    const END_TOKENS = new Set([".", "!", "?", "…", "...", PARA]);
    const RANGE_CAP = 150000;

    let N = 0;
    let surf = null;        // Int32Array: surface-form id per position
    let keys = null;        // Int32Array: lower-cased id per position (used for matching)
    let sa = null;          // Int32Array: suffix array (positions sorted by following tokens)
    let surfList = [];      // surface id -> string
    let keyList = [];       // key id -> string
    let keyIndex = new Map();
    let keyCount = null;    // Int32Array: frequency per key
    let unigram = null;     // [{id, c}] over surface ids, sorted desc

    function post(msg) { self.postMessage(msg); }

    function prep(text, keepLines) {
      text = text.replace(/\r\n?/g, "\n");
      if (keepLines) return text.replace(/\n+/g, PARA);
      return text.replace(/[ \t]*\n[ \t]*\n\s*/g, PARA).replace(/\n/g, " ");
    }

    function tokenize(text) {
      return text.match(TOKEN_RE) || [];
    }

    function build(text, keepLines) {
      const t0 = performance.now();
      post({ type: "progress", phase: "Splitting text into tokens…", pct: 5 });
      const toks = tokenize(prep(text, keepLines));
      // drop leading/duplicate paragraph marks
      const clean = [];
      for (const t of toks) {
        if (t === PARA && (clean.length === 0 || clean[clean.length - 1] === PARA)) continue;
        clean.push(t);
      }
      N = clean.length;
      surf = new Int32Array(N);
      keys = new Int32Array(N);
      surfList = [];
      keyList = [];
      keyIndex = new Map();
      const surfIndex = new Map();

      for (let i = 0; i < N; i++) {
        const t = clean[i];
        let s = surfIndex.get(t);
        if (s === undefined) { s = surfList.length; surfList.push(t); surfIndex.set(t, s); }
        surf[i] = s;
        const k = t.toLowerCase();
        let kid = keyIndex.get(k);
        if (kid === undefined) { kid = keyList.length; keyList.push(k); keyIndex.set(k, kid); }
        keys[i] = kid;
        if ((i & 0x3ffff) === 0) post({ type: "progress", phase: "Giving every token a number…", pct: 10 + 30 * i / N });
      }

      keyCount = new Int32Array(keyList.length);
      const surfCount = new Int32Array(surfList.length);
      for (let i = 0; i < N; i++) { keyCount[keys[i]]++; surfCount[surf[i]]++; }
      unigram = [];
      for (let s = 0; s < surfCount.length; s++) unigram.push({ id: s, c: surfCount[s] });
      unigram.sort((a, b) => b.c - a.c);

      post({ type: "progress", phase: "Sorting every position by what follows it (this is the slow part)…", pct: 45, indeterminate: true });
      sa = new Int32Array(N);
      for (let i = 0; i < N; i++) sa[i] = i;
      const K = keys, n = N, D = MAX_ORDER + 1;
      sa.sort((a, b) => {
        for (let j = 0; j < D; j++) {
          const ia = a + j, ib = b + j;
          if (ia >= n) return ib >= n ? 0 : -1;
          if (ib >= n) return 1;
          const d = K[ia] - K[ib];
          if (d !== 0) return d;
        }
        return 0;
      });

      // stats for the UI
      const topWords = [];
      const order = Array.from(keyCount.keys()).sort((a, b) => keyCount[b] - keyCount[a]);
      for (const kid of order) {
        if (/[\p{L}\p{N}]/u.test(keyList[kid])) topWords.push([keyList[kid], keyCount[kid]]);
        if (topWords.length >= 15) break;
      }
      let words = 0;
      for (let i = 0; i < N; i++) if (/[\p{L}\p{N}]/u.test(surfList[surf[i]])) words++;
      let distinctPairs = 0;
      for (let i = 1; i < N; i++) {
        const a = sa[i - 1], b = sa[i];
        if (keys[a] !== keys[b] || a + 1 >= N || b + 1 >= N || keys[a + 1] !== keys[b + 1]) distinctPairs++;
      }

      post({
        type: "built",
        stats: {
          tokens: N,
          words,
          vocab: keyList.length,
          pairs: distinctPairs,
          topWords,
          ms: Math.round(performance.now() - t0),
        },
      });
    }

    // Compare the tokens starting at pos with ctx. Past-the-end sorts first,
    // matching the order used when building the suffix array.
    function cmp(pos, ctx) {
      for (let j = 0; j < ctx.length; j++) {
        const p = pos + j;
        if (p >= N) return -1;
        const d = keys[p] - ctx[j];
        if (d !== 0) return d;
      }
      return 0;
    }

    function range(ctx) {
      let lo = 0, hi = N;
      while (lo < hi) { const m = (lo + hi) >> 1; if (cmp(sa[m], ctx) < 0) lo = m + 1; else hi = m; }
      const start = lo;
      hi = N;
      while (lo < hi) { const m = (lo + hi) >> 1; if (cmp(sa[m], ctx) <= 0) lo = m + 1; else hi = m; }
      return [start, lo];
    }

    // Count what follows ctx. Returns {items:[{id,c}], total} over surface ids.
    function nextCounts(ctx) {
      const k = ctx.length;
      if (k === 0) return { items: unigram, total: N, sampled: false };
      const [lo, hi] = range(ctx);
      if (hi <= lo) return null;
      const counts = new Map();
      const step = Math.max(1, Math.floor((hi - lo) / RANGE_CAP));
      let total = 0;
      for (let i = lo; i < hi; i += step) {
        const p = sa[i] + k;
        if (p >= N) continue;
        const s = surf[p];
        counts.set(s, (counts.get(s) || 0) + 1);
        total++;
      }
      if (total === 0) return null;
      const items = [];
      for (const [id, c] of counts) items.push({ id, c });
      items.sort((a, b) => b.c - a.c);
      return { items, total, sampled: step > 1, seen: hi - lo };
    }

    // Pick the longest context (up to `order` tokens) that has been seen before.
    function distribution(history, order, avoidCopy) {
      const maxK = Math.min(order, history.length);
      for (let k = maxK; k >= 1; k--) {
        const ctx = history.slice(history.length - k);
        if (ctx.some(x => x < 0)) continue;
        const res = nextCounts(ctx);
        if (!res) continue;
        if (avoidCopy && k > 1 && res.items.length < 2) continue;
        return { k, ctx, ...res };
      }
      return { k: 0, ctx: [], ...nextCounts([]) };
    }

    function applyTemperature(items, temp) {
      if (temp <= 0.001) {
        const best = items[0].c;
        const ties = items.filter(x => x.c === best);
        return items.map(x => (x.c === best ? 1 / ties.length : 0));
      }
      const lmax = Math.log(items[0].c);
      const w = items.map(x => Math.exp((Math.log(x.c) - lmax) / temp));
      const sum = w.reduce((a, b) => a + b, 0);
      return w.map(x => x / sum);
    }

    function sample(probs) {
      let r = Math.random();
      for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; }
      return probs.length - 1;
    }

    function describe(dist, probs, chosenIdx) {
      const top = [];
      const LIMIT = 10;
      for (let i = 0; i < Math.min(LIMIT, dist.items.length); i++) {
        top.push({ t: surfList[dist.items[i].id], c: dist.items[i].c, p: probs[i] });
      }
      if (chosenIdx >= LIMIT) {
        top.push({ t: surfList[dist.items[chosenIdx].id], c: dist.items[chosenIdx].c, p: probs[chosenIdx], late: true });
      }
      return {
        order: dist.k,
        ctx: dist.ctx.map(id => keyList[id]),
        options: dist.items.length,
        total: dist.total,
        sampled: !!dist.sampled,
        top,
        chosen: surfList[dist.items[chosenIdx].id],
        p: probs[chosenIdx],
      };
    }

    function promptKeys(text) {
      return tokenize(prep(text, false)).map(t => {
        const k = keyIndex.get(t.toLowerCase());
        return k === undefined ? -1 : k;
      });
    }

    function isWordKey(kid) { return /[\p{L}\p{N}]/u.test(keyList[kid]); }

    function generate(prompt, s) {
      const pk = promptKeys(prompt);
      const steps = [];
      let history;
      let how;

      // 1) Does the end of the message match the text well? Then just continue it.
      let best = 0;
      for (let k = Math.min(s.order, pk.length); k >= 2; k--) {
        const ctx = pk.slice(pk.length - k);
        if (ctx.some(x => x < 0)) continue;
        const [lo, hi] = range(ctx);
        if (hi > lo) { best = k; break; }
      }
      if (best >= 2) {
        history = pk.slice();
        how = { mode: "continue", words: best };
      } else {
        // 2) Otherwise start from the rarest word in the message that the model knows.
        let pick = -1;
        for (const k of pk) {
          if (k >= 0 && isWordKey(k) && (pick < 0 || keyCount[k] < keyCount[pick])) pick = k;
        }
        if (pick >= 0) {
          const [lo, hi] = range([pick]);
          const pos = sa[lo + Math.floor(Math.random() * (hi - lo))];
          history = [pick];
          steps.push({ t: surfList[surf[pos]], seed: true, seedWord: keyList[pick], seedCount: keyCount[pick] });
          how = { mode: "keyword", word: keyList[pick] };
        } else if (pk.length && pk[pk.length - 1] >= 0) {
          history = pk.slice();
          how = { mode: "continue", words: 1 };
        } else {
          // 3) Nothing matched: start like a new sentence.
          const dot = keyIndex.get(".");
          history = dot !== undefined ? [dot] : [];
          how = { mode: "random" };
        }
      }

      const maxTokens = [0, 30, 70, 160][s.length];
      const maxSentences = [0, 1, 3, 8][s.length];
      let sentences = 0, produced = steps.length;
      for (let guard = 0; guard < maxTokens * 3 && produced < maxTokens; guard++) {
        const dist = distribution(history, s.order, s.avoidCopy);
        if (!dist.items || dist.items.length === 0) break;
        // Only the top few hundred options matter for sampling in practice.
        if (dist.items.length > 2000) dist.items = dist.items.slice(0, 2000);
        const probs = applyTemperature(dist.items, s.temp);
        const idx = sample(probs);
        const chosenSurf = dist.items[idx].id;
        const tok = surfList[chosenSurf];
        const kid = keyIndex.get(tok.toLowerCase());
        history.push(kid);
        if (history.length > 64) history = history.slice(-MAX_ORDER);
        // don't open a reply with a blank line or stray punctuation
        if (produced === 0 && (tok === PARA || /^[.,!?;:)\]}…]/.test(tok))) continue;
        steps.push({ t: tok, info: describe(dist, probs, idx) });
        produced++;
        if (END_TOKENS.has(tok) && produced >= 3) {
          sentences++;
          if (sentences >= maxSentences) break;
        }
      }
      return { steps, how };
    }

    function predict(text, s) {
      const pk = promptKeys(text);
      const dist = distribution(pk, s.order, s.avoidCopy);
      if (!dist.items || !dist.items.length) return null;
      const items = dist.items.length > 2000 ? dist.items.slice(0, 2000) : dist.items;
      const probs = applyTemperature(items, s.temp);
      const info = describe({ ...dist, items }, probs, 0);
      info.chosen = null;
      info.unknown = tokenize(prep(text, false)).filter((t, i) => pk[i] < 0);
      return info;
    }

    self.onmessage = (e) => {
      const m = e.data;
      try {
        if (m.type === "build") build(m.text, m.keepLines);
        else if (m.type === "generate") post({ type: "generated", id: m.id, ...generate(m.prompt, m.settings) });
        else if (m.type === "predict") post({ type: "prediction", id: m.id, info: predict(m.text, m.settings) });
      } catch (err) {
        post({ type: "error", id: m.id, message: String(err && err.message || err) });
      }
    };
  }

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
    fileText: "",
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
        "\n\nIf the text is very large, your browser may have run out of memory. Try a smaller text.");
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

  function truncateWords(text, max) {
    const re = /\S+/g;
    let n = 0, m;
    while ((m = re.exec(text))) {
      if (++n === max) return text.slice(0, m.index + m[0].length);
    }
    return text;
  }

  function currentText() {
    if (state.source === "paste") return el.text.value;
    if (state.source === "upload") return state.fileText;
    const s = (window.SAMPLES || []).find(x => x.id === state.sampleId);
    return s ? s.text : "";
  }

  let counterTimer = 0;
  function updateCounter() {
    clearTimeout(counterTimer);
    counterTimer = setTimeout(() => {
      const n = countWords(currentText());
      el.counter.textContent = `${n.toLocaleString()} / ${MAX_WORDS.toLocaleString()} words` +
        (n > MAX_WORDS ? " (only the first million will be used)" : "");
      el.counter.classList.toggle("over", n > MAX_WORDS);
      state.wordCount = n;
      updateTrainButton();
    }, 150);
  }

  function updateTrainButton() {
    el.trainBtn.disabled = state.busy || !(state.wordCount > 0);
  }

  el.text.addEventListener("input", updateCounter);

  async function loadFiles(files) {
    const list = Array.from(files);
    if (!list.length) return;
    el.fileList.innerHTML = "";
    const parts = [];
    for (const f of list) {
      const li = document.createElement("li");
      li.textContent = `${f.name} (${formatBytes(f.size)})`;
      el.fileList.appendChild(li);
      try {
        parts.push(await f.text());
      } catch (err) {
        li.textContent += " could not be read";
      }
    }
    state.fileText = parts.join("\n\n");
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
    let text = currentText();
    if (!text.trim()) return;
    if (state.wordCount > MAX_WORDS) text = truncateWords(text, MAX_WORDS);
    state.busy = true;
    state.ready = false;
    updateTrainButton();
    setChatEnabled(false);
    el.stats.classList.add("hidden");
    showProgress(true);
    setProgress(2, "Starting…");
    makeWorker();
    state.worker.postMessage({ type: "build", text, keepLines: el.keepLines.checked });
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
        <div class="stat"><b>${(s.ms / 1000).toFixed(1)}s</b><span>to build</span></div>
      </div>
      <div class="top-words">Most common words: ${s.topWords.map(([w, c]) => `<code title="${c.toLocaleString()} times">${escapeHtml(w)}</code>`).join(" ")}</div>
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
  const LENGTH_NAMES = ["", "Short", "Medium", "Long"];
  function settings() {
    return {
      order: +el.order.value,
      temp: +el.temp.value / 10,
      length: +el.length.value,
      avoidCopy: el.avoidCopy.checked,
    };
  }
  function syncSettingLabels() {
    const s = settings();
    el.orderVal.textContent = s.order === 1 ? "1 word" : `${s.order} words`;
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
      if (first && /^\p{Ll}/u.test(t)) t = t[0].toUpperCase() + t.slice(1);
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
    return `<div class="ctx-line">It looked at the last <strong>${info.order} ${info.order === 1 ? "word" : "words"}</strong>:<br>${
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
    const backedOff = info.order < s.order ? `<p class="small-note">Memory is set to ${s.order} words, but that exact run of words was never seen${s.avoidCopy ? " (or had only one option)" : ""}, so it <strong>backed off</strong> to ${info.order}.</p>` : "";
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
