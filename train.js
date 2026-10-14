// UI for train.html. The neural network itself is in nn-model.js and runs in a Web Worker.
(() => {
  "use strict";

  const MAX_CHARS = 20_000_000;
  const LAYERS = [1, 2, 3, 4, 6, 8];
  const WIDTHS = [32, 48, 64, 96, 128, 192, 256];
  const HEADS = [1, 2, 4, 6, 8];
  const CONTEXTS = [16, 32, 64, 96, 128, 256];
  const TOPK = [0, 1, 3, 5, 10, 20, 50];
  const LENGTHS = [0, 25, 50, 100, 200, 300, 500, 750, 1000];
  const ARCH_PRESETS = [
    { name: "Tiny", note: "fast: a few minutes", L: 2, C: 64, NH: 4, T: 64, steps: 600, B: 16, lr: 0.003 },
    { name: "Small", note: "better: about 3× slower", L: 3, C: 96, NH: 4, T: 96, steps: 1000, B: 16, lr: 0.002 },
    { name: "Medium", note: "best: leave it running", L: 4, C: 128, NH: 4, T: 128, steps: 2000, B: 16, lr: 0.002 },
  ];
  const STUDENT_PRESETS = [
    { name: "Mini", note: "1 layer, width 32", L: 1, C: 32, NH: 2 },
    { name: "Small", note: "1 layer, width 64", L: 1, C: 64, NH: 4 },
    { name: "Deep & thin", note: "2 layers, width 32", L: 2, C: 32, NH: 2 },
  ];

  const $ = (id) => document.getElementById(id);
  const el = {
    tabs: document.querySelectorAll(".tab"), panels: document.querySelectorAll(".tab-panel"),
    text: $("text-input"), file: $("file-input"), drop: $("dropzone"), fileList: $("file-list"), samples: $("samples"),
    tokKind: $("tok-kind"), vocabSize: $("vocab-size"), vocabField: $("vocab-field"),
    counter: $("char-counter"), reuseWrap: $("reuse-wrap"), reuse: $("reuse-tok"), tokBtn: $("tokenize-btn"),
    tokProgress: $("tok-progress"), tokFill: $("tok-progress-fill"), tokLabel: $("tok-progress-label"), tokStats: $("tok-stats"),
    archCard: $("arch-card"), archPresets: $("arch-presets"), aLayers: $("a-layers"), aWidth: $("a-width"), aHeads: $("a-heads"), aCtx: $("a-ctx"),
    archSummary: $("arch-summary"), createBtn: $("create-btn"), createStats: $("create-stats"),
    trainCard: $("train-card"), tSteps: $("t-steps"), tBatch: $("t-batch"), tLr: $("t-lr"), tWarmup: $("t-warmup"), tWd: $("t-wd"),
    trainBtn: $("train-btn"), pauseBtn: $("pause-btn"), trainEta: $("train-eta"),
    trainProgress: $("train-progress"), trainFill: $("train-progress-fill"), trainLabel: $("train-progress-label"),
    lossChart: $("loss-chart"), lossLegend: $("loss-legend"), trainTiles: $("train-tiles"), compress: $("compress-box"),
    anaInput: $("analysis-input"), anaWhich: $("analysis-which"), anaBtn: $("analysis-btn"), anaSummary: $("analysis-summary"),
    anaOut: $("analysis-out"), anaDetail: $("analysis-detail"),
    genCard: $("gen-card"), genOut: $("gen-out"), genForm: $("gen-form"), genInput: $("gen-input"), genBtn: $("gen-btn"),
    genStop: $("gen-stop"), genClear: $("gen-clear"), genWhich: $("gen-which"), genTemp: $("gen-temp"), genTempVal: $("gen-temp-val"),
    genTopk: $("gen-topk"), genTopkVal: $("gen-topk-val"), genLen: $("gen-len"), genLenVal: $("gen-len-val"), genInspector: $("gen-inspector"),
    distillCard: $("distill-card"), studentPresets: $("student-presets"), sLayers: $("s-layers"), sWidth: $("s-width"), sHeads: $("s-heads"),
    dSteps: $("d-steps"), dTau: $("d-tau"), dArch: $("d-arch-summary"), distillBtn: $("distill-btn"), dpauseBtn: $("dpause-btn"),
    distillEta: $("distill-eta"), distillProgress: $("distill-progress"), distillFill: $("distill-progress-fill"), distillLabel: $("distill-progress-label"),
    distillChart: $("distill-chart"), distillLegend: $("distill-legend"), distillTiles: $("distill-tiles"), distillRuns: $("distill-runs"),
    saveCard: $("save-card"), saveTeacher: $("save-teacher"), saveStudent: $("save-student"), loadModel: $("load-model"), saveStatus: $("save-status"),
  };

  const state = {
    source: "paste", files: [], sampleId: null, chars: 0,
    worker: null, reqId: 0, pending: new Map(),
    tok: null,            // stats from the tokenizer
    hasData: false,       // the worker holds tokenized training text
    model: null,          // info about the pre-trained (teacher) model
    student: null,        // info about the distilled student
    tokenizing: false, training: false, distilling: false, generating: false,
    trainPaused: false, distillPaused: false, distillKey: "",
    th: newHistory(), dh: newHistory(),
    runs: [],
    genTarget: null,
  };

  function newHistory() { return { train: [], ema: [], val: [], ce: [], kl: [], h: [], emaV: 0, emaN: 0, msEma: 0, last: null, teacherVal: null }; }

  // ---------------------------------------------------------------------------
  // Worker plumbing
  // ---------------------------------------------------------------------------
  const FINAL = new Set(["tokenized", "created", "tdone", "ddone", "gendone", "analysis", "prediction", "exported", "imported"]);

  function makeWorker() {
    const src = `(${nnWorkerMain.toString()})();`;
    const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
    const w = new Worker(url);
    URL.revokeObjectURL(url);
    w.onmessage = onMessage;
    w.onerror = (e) => {
      alert("The worker crashed: " + (e.message || "unknown error") + "\n\nThe model may be too big for your browser's memory. Try a smaller model or batch.");
      state.tokenizing = state.training = state.distilling = state.generating = false;
      refreshButtons();
    };
    state.worker = w;
  }

  function call(msg, transfer) {
    return new Promise((resolve, reject) => {
      const id = ++state.reqId;
      state.pending.set(id, { resolve, reject });
      state.worker.postMessage({ ...msg, id }, transfer || []);
    });
  }

  function onMessage(e) {
    const m = e.data;
    switch (m.type) {
      case "progress": setBar(el.tokFill, el.tokLabel, m.pct, m.phase); break;
      case "tstart": onTrainStart(m); break;
      case "tstep": onStep(state.th, m, false); break;
      case "teval": onEval(state.th, m, false); break;
      case "dstart": onDistillStart(m); break;
      case "dstep": onStep(state.dh, m, true); break;
      case "deval": onEval(state.dh, m, true); break;
      case "gen": onGenToken(m); break;
    }
    if (m.id && state.pending.has(m.id) && (FINAL.has(m.type) || m.type === "error")) {
      const p = state.pending.get(m.id);
      state.pending.delete(m.id);
      if (m.type === "error") p.reject(new Error(m.message)); else p.resolve(m);
    }
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  // Makes spaces and line breaks visible inside a token.
  function showTok(s) { return s.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥"); }
  function fmt(n, d = 2) { return n == null || !isFinite(n) ? "–" : n.toFixed(d); }
  function fmtInt(n) { return Math.round(n).toLocaleString(); }
  function fmtP(p) {
    if (p >= 0.995) return "100%";
    if (p >= 0.1) return Math.round(p * 100) + "%";
    if (p >= 0.01) return (p * 100).toFixed(1) + "%";
    if (p >= 0.001) return (p * 100).toFixed(2) + "%";
    return p > 0 ? "<0.1%" : "0%";
  }
  function fmtBytes(b) {
    if (b < 1024) return Math.round(b) + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / 1024 / 1024).toFixed(2) + " MB";
  }
  function fmtTime(ms) {
    const s = Math.round(ms / 1000);
    if (s < 60) return s + " s";
    if (s < 3600) return Math.floor(s / 60) + " min " + (s % 60) + " s";
    return Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min";
  }
  function setBar(fill, label, pct, text) {
    fill.style.width = Math.max(0, Math.min(100, pct)) + "%";
    if (text != null) label.textContent = text;
  }
  function show(node, on) { node.classList.toggle("hidden", !on); }
  function enableCard(card, on) { card.classList.toggle("disabled", !on); }
  function probClass(p) { return p >= 0.8 ? "p-hi" : p >= 0.25 ? "p-mid" : "p-lo"; }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function fillSelect(sel, values, label, value) {
    sel.innerHTML = values.map(v => `<option value="${v}">${label ? label(v) : v}</option>`).join("");
    if (value != null) sel.value = value;
  }

  function paramCount(c) {
    return c.V * c.C + c.T * c.C + c.L * (12 * c.C * c.C + 13 * c.C) + 2 * c.C;
  }

  function barsHtml(top, chosen) {
    const maxP = Math.max(...top.map(x => x.p), 1e-9);
    return `<div class="bars">${top.map(o => `
      <div class="bar-row${o.s === chosen ? " chosen" : ""}">
        <span class="w" title="${escapeHtml(o.s)}">${escapeHtml(showTok(o.s))}</span>
        <span class="track"><span class="fill" style="width:${(100 * o.p / maxP).toFixed(1)}%; display:block"></span></span>
        <span class="p">${fmtP(o.p)}</span>
      </div>`).join("")}</div>`;
  }

  // ---------------------------------------------------------------------------
  // Step 1: text and tokenizer
  // ---------------------------------------------------------------------------
  el.tabs.forEach(tab => tab.addEventListener("click", () => {
    el.tabs.forEach(t => { t.classList.toggle("active", t === tab); t.setAttribute("aria-selected", t === tab); });
    el.panels.forEach(p => p.classList.toggle("hidden", p.dataset.panel !== tab.dataset.tab));
    state.source = tab.dataset.tab;
    updateCounter();
  }));

  const ALL_SAMPLES = { id: "__all", title: "All the samples together", blurb: "Every built-in sample, joined." };
  const samples = window.SAMPLES || [];
  function sampleText(id) {
    if (id === ALL_SAMPLES.id) return samples.map(s => s.text).join("\n\n");
    const s = samples.find(x => x.id === id);
    return s ? s.text : "";
  }
  [...samples, ALL_SAMPLES].forEach(s => {
    const b = document.createElement("button");
    b.className = "sample";
    b.type = "button";
    b.innerHTML = `<strong></strong><span></span>`;
    b.querySelector("strong").textContent = s.title;
    b.querySelector("span").textContent = `${s.blurb} (${sampleText(s.id).length.toLocaleString()} characters)`;
    b.addEventListener("click", () => {
      state.sampleId = s.id;
      el.samples.querySelectorAll(".sample").forEach(x => x.classList.toggle("active", x === b));
      updateCounter();
    });
    el.samples.appendChild(b);
  });

  function loadFiles(files) {
    const list = Array.from(files);
    if (!list.length) return;
    state.files = list;
    el.fileList.innerHTML = "";
    for (const f of list) {
      const li = document.createElement("li");
      li.textContent = `${f.name} (${fmtBytes(f.size)})`;
      el.fileList.appendChild(li);
    }
    updateCounter();
  }
  el.file.addEventListener("change", () => loadFiles(el.file.files));
  ["dragenter", "dragover"].forEach(ev => el.drop.addEventListener(ev, e => { e.preventDefault(); el.drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => el.drop.addEventListener(ev, e => { e.preventDefault(); el.drop.classList.remove("drag"); }));
  el.drop.addEventListener("drop", e => loadFiles(e.dataTransfer.files));
  el.text.addEventListener("input", updateCounter);

  function updateCounter() {
    let n, about = "";
    if (state.source === "upload") { n = state.files.reduce((a, f) => a + f.size, 0); about = n ? "about " : ""; }
    else if (state.source === "sample") n = sampleText(state.sampleId).length;
    else n = el.text.value.length;
    state.chars = n;
    el.counter.textContent = `${about}${n.toLocaleString()} characters` + (n > MAX_CHARS ? ` (only the first ${MAX_CHARS.toLocaleString()} will be used)` : "");
    el.counter.classList.toggle("over", n > MAX_CHARS);
    refreshButtons();
  }

  async function currentText() {
    if (state.source === "paste") return el.text.value;
    if (state.source === "sample") return sampleText(state.sampleId);
    let out = "";
    for (const f of state.files) {
      if (out.length >= MAX_CHARS) break;
      out += (out ? "\n\n" : "") + await f.text();
    }
    return out;
  }

  el.tokKind.addEventListener("change", () => show(el.vocabField, el.tokKind.value === "bpe"));

  el.tokBtn.addEventListener("click", async () => {
    const reuse = state.model && el.reuse.checked;
    if (state.model && !reuse && !confirm("Tokenizing again makes a new tokenizer, so the current model will be deleted. Continue?")) return;
    let text = await currentText();
    if (!text.trim()) return;
    if (text.length > MAX_CHARS) text = text.slice(0, MAX_CHARS);
    state.tokenizing = true;
    refreshButtons();
    show(el.tokProgress, true);
    setBar(el.tokFill, el.tokLabel, 0, "Starting…");
    try {
      const res = await call({ type: "tokenize", text, kind: el.tokKind.value, vocabSize: +el.vocabSize.value, reuse });
      state.tok = res.stats;
      state.hasData = true;
      if (!res.stats.reused) resetModel();
      renderTokStats(res.stats);
      setBar(el.tokFill, el.tokLabel, 100, `Done in ${(res.stats.ms / 1000).toFixed(1)} s.`);
      setTimeout(() => show(el.tokProgress, false), 800);
      updateArchSummary();
      if (state.model) { renderCompression(); drawSoon(); }
      else el.archCard.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      show(el.tokProgress, false);
      alert("Could not tokenize: " + err.message);
    }
    state.tokenizing = false;
    refreshButtons();
  });

  function renderTokStats(s) {
    const cpt = s.chars / s.tokens;
    const gzBitsTok = s.gzipBytes != null ? s.gzipBytes * 8 / s.valTokens : null;
    el.tokStats.innerHTML = `
      <div class="stat-row">
        <div class="stat"><b>${s.chars.toLocaleString()}</b><span>characters</span></div>
        <div class="stat"><b>${s.tokens.toLocaleString()}</b><span>tokens</span></div>
        <div class="stat"><b>${s.vocab.toLocaleString()}</b><span>vocabulary size${s.kind === "bpe" ? ` (${s.merges.toLocaleString()} learned merges)` : ""}</span></div>
        <div class="stat"><b>${cpt.toFixed(2)}</b><span>characters per token</span></div>
        <div class="stat"><b>${s.trainTokens.toLocaleString()} / ${s.valTokens.toLocaleString()}</b><span>training / validation tokens (90% / 10%)</span></div>
      </div>
      <p class="note">${s.reused ? "<strong>Kept the model's tokenizer.</strong> " : ""}The first tokens of your text (each colour is one token, <code>·</code> is a space, <code>↵</code> a new line):</p>
      <div class="token-chips">${s.preview.map(t => `<span class="chip" title="${escapeHtml(JSON.stringify(t))}">${escapeHtml(t.replace(/\n/g, "↵"))}</span>`).join("")}</div>
      ${s.kind === "bpe" && s.longest.length ? `<div class="top-words">Longest learned tokens: ${s.longest.map(t => `<code>${escapeHtml(showTok(t))}</code>`).join(" ")}</div>` : ""}
      <div class="top-words">Most common tokens: ${s.common.map(([t, c]) => `<code title="${c.toLocaleString()} times">${escapeHtml(showTok(t))}</code>`).join(" ")}</div>
      ${s.dropped ? `<p class="note muted">${s.dropped.toLocaleString()} rare characters were not given their own token. They become <code>�</code>.</p>` : ""}
      <p class="note muted">Baselines on the validation text, in bits per token: guessing uniformly costs log₂(${s.vocab.toLocaleString()}) = <strong>${fmt(s.uniformBits)}</strong>. A model that only knows how common each token is (no context) costs <strong>${fmt(s.unigramBits)}</strong>.${gzBitsTok != null ? ` gzip costs <strong>${fmt(gzBitsTok)}</strong>.` : ""} Your network should beat all of these.</p>
      ${s.tokens < 50000 ? `<p class="note"><strong>This is a small text.</strong> A neural network will soon memorise it: the training loss keeps falling while the validation loss stops falling or goes up. Upload a book for much better results.</p>` : ""}`;
    show(el.tokStats, true);
  }

  // ---------------------------------------------------------------------------
  // Step 2: architecture
  // ---------------------------------------------------------------------------
  fillSelect(el.aLayers, LAYERS, null, 2);
  fillSelect(el.aWidth, WIDTHS, null, 64);
  fillSelect(el.aHeads, HEADS, null, 4);
  fillSelect(el.aCtx, CONTEXTS, null, 64);

  function archCfg() {
    return { L: +el.aLayers.value, C: +el.aWidth.value, NH: +el.aHeads.value, T: +el.aCtx.value };
  }

  function syncHeads(widthSel, headSel) {
    const C = +widthSel.value;
    [...headSel.options].forEach(o => { o.disabled = C % +o.value !== 0; });
    if (headSel.selectedOptions[0].disabled) headSel.value = [...headSel.options].reverse().find(o => !o.disabled).value;
  }

  ARCH_PRESETS.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset" + (i === 0 ? " active" : "");
    b.innerHTML = `<strong>${p.name}</strong><span>${p.L} layers · width ${p.C} · context ${p.T} · ${p.note}</span>`;
    b.addEventListener("click", () => {
      el.archPresets.querySelectorAll(".preset").forEach(x => x.classList.toggle("active", x === b));
      el.aLayers.value = p.L; el.aWidth.value = p.C; el.aHeads.value = p.NH; el.aCtx.value = p.T;
      el.tSteps.value = p.steps; el.tBatch.value = p.B; el.tLr.value = p.lr;
      syncHeads(el.aWidth, el.aHeads);
      updateArchSummary();
    });
    el.archPresets.appendChild(b);
  });
  [el.aLayers, el.aWidth, el.aHeads, el.aCtx].forEach(s => s.addEventListener("change", () => {
    el.archPresets.querySelectorAll(".preset").forEach(x => x.classList.remove("active"));
    syncHeads(el.aWidth, el.aHeads);
    updateArchSummary();
  }));

  function archDiagram(c, V) {
    return `<div class="diagram">
      <span class="blk">tokens</span>→
      <span class="blk">embed ${V}×${c.C} + position ${c.T}×${c.C}</span>→
      <span class="blk hl">[ LayerNorm → attention (${c.NH} heads) → + → LayerNorm → MLP ${c.C}→${4 * c.C}→${c.C} → + ] × ${c.L}</span>→
      <span class="blk">LayerNorm</span>→
      <span class="blk">output ${c.C}×${V}</span>→
      <span class="blk">softmax: ${V} probabilities</span>
    </div>`;
  }

  function updateArchSummary() {
    const c = archCfg();
    const V = state.tok ? state.tok.vocab : +el.vocabSize.value;
    const n = paramCount({ ...c, V });
    el.archSummary.innerHTML = `This network has <strong>${n.toLocaleString()}</strong> weights (GPT-2 small has 124 million; big models have hundreds of billions).${archDiagram(c, V)}`;
    updateStudentSummary();
  }

  el.createBtn.addEventListener("click", async () => {
    if (state.model && !confirm("This replaces the current model with a new, untrained one. Continue?")) return;
    const cfg = archCfg();
    if (state.tok && state.tok.valTokens < cfg.T + 2) { alert("The validation text is shorter than the context. Pick a shorter context or use more text."); return; }
    el.createBtn.disabled = true;
    try {
      const res = await call({ type: "create", cfg });
      resetModel();
      state.model = res.info;
      state.th.val.push([0, res.initLoss]);
      el.createStats.innerHTML = `
        <div class="stat-row">
          <div class="stat"><b>${res.info.params.toLocaleString()}</b><span>weights, all small random numbers</span></div>
          <div class="stat"><b>${fmt(res.initLoss)}</b><span>bits per token: its loss before training</span></div>
          <div class="stat"><b>${fmt(state.tok.uniformBits)}</b><span>bits per token for a uniform guess, log₂(${state.tok.vocab.toLocaleString()})</span></div>
        </div>
        <p class="note muted">With random weights, the model gives every token about the same chance, so its loss is close to log₂(vocabulary size). In other words, it can't compress the text at all yet. Training will push this number down.</p>`;
      show(el.createStats, true);
      renderTrainTiles();
      renderCompression();
      drawSoon();
      el.trainCard.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      alert("Could not create the model: " + err.message);
    }
    refreshButtons();
  });

  function resetModel() {
    state.model = null; state.student = null;
    state.th = newHistory(); state.dh = newHistory();
    state.trainPaused = state.distillPaused = false;
    state.runs = [];
    el.distillRuns.innerHTML = ""; show(el.distillRuns, false);
    show(el.createStats, false);
    el.trainBtn.textContent = "Start training";
    el.distillBtn.textContent = "Start distilling";
    el.trainEta.textContent = el.distillEta.textContent = "";
    setStudentOptions(false);
    renderTrainTiles(); renderDistillTiles();
    show(el.compress, false);
    drawSoon();
  }

  // ---------------------------------------------------------------------------
  // Step 3: pre-training
  // ---------------------------------------------------------------------------
  function trainOpts(steps) {
    const B = +el.tBatch.value;
    return {
      steps: Math.max(1, Math.floor(steps)), B, lr: +el.tLr.value, warmup: Math.max(0, +el.tWarmup.value || 0),
      wd: +el.tWd.value, evalEvery: Math.max(10, Math.round(steps / 40)), evalRows: Math.max(B, 32),
    };
  }

  el.trainBtn.addEventListener("click", async () => {
    const resume = state.trainPaused;
    state.training = true;
    refreshButtons();
    try {
      const res = await call({ type: "train", opts: trainOpts(+el.tSteps.value || 1), resume });
      state.model = res.info;
      state.trainPaused = !res.finished;
      el.trainBtn.textContent = res.finished ? "Train more" : "Continue training";
      el.trainLabel.textContent = res.finished ? `Finished: ${res.info.step.toLocaleString()} steps in total.` : "Paused.";
    } catch (err) {
      alert("Training stopped: " + err.message);
    }
    state.training = false;
    el.trainEta.textContent = "";
    refreshButtons();
  });
  el.pauseBtn.addEventListener("click", () => state.worker.postMessage({ type: "stop" }));

  function onTrainStart(m) {
    state.model = m.info;
    show(el.trainProgress, true);
  }

  function onStep(h, m, distill) {
    const owner = distill ? state.student : state.model;
    if (owner) owner.step = m.step;
    h.train.push([m.step, m.loss]);
    h.emaN++;
    h.emaV = 0.9 * h.emaV + 0.1 * m.loss;
    h.ema.push([m.step, h.emaV / (1 - Math.pow(0.9, h.emaN))]);
    if (m.ce != null) { h.ce.push([m.step, m.ce]); h.kl.push([m.step, m.kl]); h.h.push([m.step, m.h]); }
    h.msEma = h.msEma ? 0.8 * h.msEma + 0.2 * m.ms : m.ms;
    h.last = m;
    const eta = (m.total - m.done) * h.msEma;
    const fill = distill ? el.distillFill : el.trainFill, label = distill ? el.distillLabel : el.trainLabel;
    setBar(fill, label, 100 * m.done / m.total, `Step ${m.done.toLocaleString()} of ${m.total.toLocaleString()} · ${fmtInt(m.tokPerSec)} tokens/s · ${Math.round(h.msEma)} ms per step`);
    (distill ? el.distillEta : el.trainEta).textContent = m.done < m.total ? `about ${fmtTime(eta)} left` : "";
    if (distill) renderDistillTiles(); else renderTrainTiles();
    drawSoon();
  }

  function onEval(h, m, distill) {
    h.val.push([m.step, m.val]);
    if (m.teacherVal != null) h.teacherVal = m.teacherVal;
    if (distill) {
      if (state.student) state.student.val = m.val;
      renderDistillTiles();
    } else {
      renderTrainTiles();
    }
    renderCompression();
    drawSoon();
  }

  function renderTrainTiles() {
    const h = state.th, last = h.last, val = h.val.length ? h.val[h.val.length - 1][1] : null;
    const cpt = state.tok ? state.tok.chars / state.tok.tokens : 1;
    const tile = (big, label, sub) => `<div class="stat"><b>${big}</b><span>${label}</span>${sub ? `<span class="sub">${sub}</span>` : ""}</div>`;
    el.trainTiles.innerHTML =
      tile(state.model ? state.model.step.toLocaleString() : "–", "steps trained", last ? `${(state.model.step * (+el.tBatch.value) * state.model.cfg.T).toLocaleString()} tokens seen` : "") +
      tile(last ? fmt(h.ema[h.ema.length - 1][1]) : "–", "training loss (bits per token)", last ? `= ${fmt(h.ema[h.ema.length - 1][1] * Math.LN2)} nats (smoothed)` : "") +
      tile(val != null ? fmt(val) : "–", "validation loss (bits per token)", val != null ? `text it never trained on · ${fmt(val / cpt)} bits per character` : "") +
      tile(val != null ? fmtP(Math.pow(2, -val)) : "–", "average chance of the right token", val != null ? `perplexity ${fmt(Math.pow(2, val), 1)}: as unsure as choosing between ${fmt(Math.pow(2, val), 1)} tokens` : "") +
      tile(last ? last.lr.toExponential(1) : "–", "learning rate now", last ? `gradient size ${fmt(last.gnorm)}` : "");
  }

  // How many bytes each method needs for the validation text.
  function renderCompression() {
    const s = state.tok;
    if (!s || !state.model) { show(el.compress, false); return; }
    const val = state.th.val.length ? state.th.val[state.th.val.length - 1][1] : null;
    const rows = [
      { name: "Plain text (UTF-8, about 8 bits per character)", bytes: s.valChars },
      { name: `Uniform guess over ${s.vocab.toLocaleString()} tokens`, bytes: s.uniformBits * s.valTokens / 8 },
      { name: "Token counts only (no context)", bytes: s.unigramBits * s.valTokens / 8 },
    ];
    if (s.gzipBytes != null) {
      rows.push({ name: "gzip", bytes: s.gzipBytes });
      rows.push({ name: "gzip, after reading the end of the training text (the zipping trick)", bytes: s.gzipCondBytes });
    }
    if (val != null) rows.push({ name: "Your neural network (validation loss)", bytes: val * s.valTokens / 8, model: true });
    const sv = state.dh.val.length ? state.dh.val[state.dh.val.length - 1][1] : null;
    if (state.student && sv != null) rows.push({ name: "Your distilled student", bytes: sv * s.valTokens / 8, model: true });
    const max = Math.max(...rows.map(r => r.bytes));
    const best = Math.min(...rows.map(r => r.bytes));
    el.compress.innerHTML = `
      <h3>Your model is a compressor</h3>
      <p class="setting-help">The validation text has ${s.valChars.toLocaleString()} characters. A model that gives probability p to each real next token could store it in Σ −log₂ p bits (using arithmetic coding). So: lower loss = smaller file.</p>
      <table>
        <tr><th>Method</th><th>bits / token</th><th>bits / char</th><th>size</th><th style="width:30%"></th></tr>
        ${rows.map(r => `<tr class="${r.model ? "model" : ""}${r.bytes === best ? " win" : ""}">
          <td>${r.name}${r.bytes === best ? " 🏆" : ""}</td>
          <td class="num">${fmt(r.bytes * 8 / s.valTokens)}</td>
          <td class="num">${fmt(r.bytes * 8 / s.valChars)}</td>
          <td class="num">${fmtBytes(r.bytes)}</td>
          <td><div class="bar" style="width:${(100 * r.bytes / max).toFixed(1)}%"></div></td></tr>`).join("")}
      </table>`;
    show(el.compress, true);
  }

  // ---------------------------------------------------------------------------
  // Charts
  // ---------------------------------------------------------------------------
  let drawPending = false;
  function drawSoon() {
    if (drawPending) return;
    drawPending = true;
    requestAnimationFrame(() => { drawPending = false; drawTrainChart(); drawDistillChart(); });
  }

  function drawChart(canvas, opts) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 44, r: 12, t: 12, b: 30 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    let yMax = opts.yMax, xMax = Math.max(1, opts.xMax);
    const X = x => pad.l + pw * x / xMax, Y = y => pad.t + ph * (1 - Math.min(y, yMax) / yMax);
    const muted = cssVar("--muted"), border = cssVar("--border");
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = muted; ctx.strokeStyle = border; ctx.lineWidth = 1;
    const yStep = yMax > 12 ? 2 : yMax > 5 ? 1 : 0.5;
    for (let y = 0; y <= yMax + 1e-9; y += yStep) {
      ctx.beginPath(); ctx.moveTo(pad.l, Y(y)); ctx.lineTo(pad.l + pw, Y(y)); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(y.toString(), pad.l - 6, Y(y));
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const xStep = niceStep(xMax / 6);
    for (let x = 0; x <= xMax; x += xStep) ctx.fillText(x.toLocaleString(), X(x), pad.t + ph + 6);
    ctx.save();
    ctx.translate(11, pad.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle";
    ctx.fillText(opts.yLabel, 0, 0); ctx.restore();
    ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText(opts.xLabel, pad.l + pw, pad.t + ph + 28);
    for (const hl of opts.hlines || []) {
      if (hl.y == null || !isFinite(hl.y)) continue;
      ctx.strokeStyle = hl.color; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(hl.y)); ctx.lineTo(pad.l + pw, Y(hl.y)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = hl.color; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText(hl.label, pad.l + 4, Y(hl.y) - 2);
    }
    for (const s of opts.series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.globalAlpha = s.alpha || 1;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      s.points.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)));
      ctx.stroke();
      if (s.dots) {
        ctx.fillStyle = s.color;
        for (const [x, y] of s.points) { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, 2 * Math.PI); ctx.fill(); }
      }
      ctx.globalAlpha = 1; ctx.setLineDash([]);
    }
  }
  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(1, raw))));
    for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
    return 10 * p;
  }
  function legend(node, items) {
    node.innerHTML = items.map(i => `<span><i class="${i.dash ? "dash" : ""}" style="background:${i.color};border-color:${i.color}"></i>${i.label}</span>`).join("");
  }

  function drawTrainChart() {
    const h = state.th, s = state.tok;
    const cTrain = cssVar("--line-train"), cVal = cssVar("--line-val"), cBase = cssVar("--line-base"), cGreen = cssVar("--line-teacher");
    const all = [...h.ema, ...h.val].map(p => p[1]);
    const yMax = Math.ceil(Math.max(s ? s.uniformBits : 4, ...all, 1) + 0.5);
    const xMax = Math.max(state.model ? state.model.step : 0, +el.tSteps.value || 0, ...h.train.map(p => p[0]));
    const gz = s && s.gzipBytes != null ? s.gzipCondBytes * 8 / s.valTokens : null;
    drawChart(el.lossChart, {
      yMax, xMax, yLabel: "loss (bits per token)", xLabel: "training step",
      hlines: s ? [
        { y: s.uniformBits, color: cBase, label: "uniform guess" },
        { y: s.unigramBits, color: cBase, label: "token counts (no context)" },
        { y: gz, color: cGreen, label: "gzip" },
      ] : [],
      series: [
        { points: h.train, color: cTrain, width: 1, alpha: 0.25 },
        { points: h.ema, color: cTrain, width: 2 },
        { points: h.val, color: cVal, width: 2, dots: h.val.length < 60 },
      ],
    });
    legend(el.lossLegend, [
      { color: cTrain, label: "training loss (on the batch it is learning from)" },
      { color: cVal, label: "validation loss (on text it never trains on)" },
      { color: cGreen, label: "gzip", dash: true },
      { color: cBase, label: "baselines", dash: true },
    ]);
  }

  function drawDistillChart() {
    const h = state.dh, s = state.tok;
    const cTrain = cssVar("--line-train"), cVal = cssVar("--line-val"), cT = cssVar("--line-teacher"), cBase = cssVar("--line-base");
    const all = [...h.ema, ...h.val, ...h.ce, ...h.kl].map(p => p[1]);
    const yMax = Math.ceil(Math.max(s ? s.uniformBits : 4, ...all, 1) + 0.5);
    const xMax = Math.max(state.student ? state.student.step : 0, +el.dSteps.value || 0, ...h.train.map(p => p[0]));
    drawChart(el.distillChart, {
      yMax, xMax, yLabel: "bits per token", xLabel: "distillation step",
      hlines: [{ y: h.teacherVal, color: cT, label: "teacher's validation loss" }],
      series: [
        { points: h.ce, color: cTrain, width: 2 },
        { points: h.kl, color: cTrain, width: 2, dash: [6, 4] },
        { points: h.h, color: cBase, width: 1.5 },
        { points: h.ema, color: cVal, width: 1, alpha: 0.5 },
        { points: h.val, color: cVal, width: 2, dots: h.val.length < 60 },
      ],
    });
    legend(el.distillLegend, [
      { color: cTrain, label: "cross-entropy H(teacher, student)" },
      { color: cTrain, label: "KL divergence = cross-entropy − entropy", dash: true },
      { color: cBase, label: "teacher's entropy H(teacher)" },
      { color: cVal, label: "student's loss on the real next tokens (faint: training, bold: validation)" },
      { color: cT, label: "teacher's validation loss", dash: true },
    ]);
  }
  window.addEventListener("resize", drawSoon);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawSoon);

  // ---------------------------------------------------------------------------
  // Token-by-token loss
  // ---------------------------------------------------------------------------
  el.anaBtn.addEventListener("click", async () => {
    el.anaBtn.disabled = true;
    el.anaSummary.textContent = "Scoring…";
    try {
      const res = await call({ type: "analyze", which: el.anaWhich.value, text: el.anaInput.value });
      renderAnalysis(res);
    } catch (err) {
      el.anaSummary.textContent = "Error: " + err.message;
    }
    refreshButtons();
  });

  function heatStyle(bits) {
    // 0–1 bits: expected (cool). Above 4 bits: surprising (warm).
    if (bits == null) return "";
    if (bits < 1) return `background: color-mix(in srgb, var(--cool) ${Math.round(100 * (1 - bits))}%, transparent)`;
    if (bits > 3) return `background: color-mix(in srgb, var(--warm) ${Math.round(Math.min(100, (bits - 3) * 18))}%, transparent)`;
    return "";
  }

  function renderAnalysis(res) {
    el.anaSummary.innerHTML = `Average: <strong>${fmt(res.avgBits)} bits per token</strong> (${fmt(res.bitsPerChar)} bits per character) over ${(res.tokens.length - 1).toLocaleString()} tokens. That average is the loss.`;
    el.anaOut.innerHTML = "";
    res.tokens.forEach((t, i) => {
      const span = document.createElement("span");
      span.className = "st" + (t.first ? " first" : "");
      span.textContent = t.s === "\n" ? "↵\n" : t.s;
      if (!t.first) {
        span.style.cssText = heatStyle(t.bits);
        span.title = `p = ${fmtP(t.p)} · ${fmt(t.bits)} bits`;
      }
      span.addEventListener("click", () => {
        el.anaOut.querySelectorAll(".selected").forEach(x => x.classList.remove("selected"));
        span.classList.add("selected");
        if (t.first) { el.anaDetail.className = ""; el.anaDetail.innerHTML = `<p class="small-note">The first token has nothing before it, so it is not scored.</p>`; return; }
        el.anaDetail.className = "";
        el.anaDetail.innerHTML = `<div class="ctx-line">The real next token was <code>${escapeHtml(showTok(t.s))}</code>. Before seeing it, the model gave it <strong>${fmtP(t.p)}</strong>, so its information is −log₂ ${t.p < 0.001 ? t.p.toExponential(1) : t.p.toFixed(3)} = <strong>${fmt(t.bits)} bits</strong>.</div>
          <div class="ctx-line">The model's top guesses were:</div>${barsHtml(t.top, t.s)}`;
      });
      el.anaOut.appendChild(span);
    });
    const legendNode = document.createElement("div");
    legendNode.className = "heat-legend";
    legendNode.innerHTML = `<span>expected (0 bits)</span><span class="ramp"></span><span>surprising (8+ bits)</span>`;
    el.anaOut.appendChild(legendNode);
  }

  // ---------------------------------------------------------------------------
  // Step 4: generation
  // ---------------------------------------------------------------------------
  function genSettings() {
    return { temp: +el.genTemp.value / 10, topK: TOPK[+el.genTopk.value], n: LENGTHS[+el.genLen.value] };
  }
  function syncGenLabels() {
    const s = genSettings();
    el.genTempVal.textContent = s.temp === 0 ? "0 (always top)" : s.temp.toFixed(1);
    el.genTopkVal.textContent = s.topK ? s.topK : "off";
    el.genLenVal.textContent = `${s.n} tokens`;
  }
  [el.genTemp, el.genTopk, el.genLen].forEach(x => x.addEventListener("input", syncGenLabels));
  syncGenLabels();

  el.genForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state.generating || !state.model) return;
    const prompt = el.genInput.value;
    state.generating = true;
    refreshButtons();
    el.genOut.innerHTML = "";
    const p = document.createElement("span");
    p.className = "prompt-text";
    p.textContent = prompt;
    el.genOut.appendChild(p);
    state.genTarget = el.genOut;
    try {
      await call({ type: "generate", prompt, which: el.genWhich.value, ...genSettings() });
    } catch (err) {
      el.genOut.appendChild(document.createTextNode("\n[Error: " + err.message + "]"));
    }
    state.generating = false;
    refreshButtons();
  });
  el.genStop.addEventListener("click", () => state.worker.postMessage({ type: "stopgen" }));
  el.genClear.addEventListener("click", () => {
    el.genOut.innerHTML = "";
    el.genInspector.className = "inspector-empty";
    el.genInspector.textContent = "Click a generated token to see the probability distribution the model gave.";
  });

  function onGenToken(m) {
    if (!state.genTarget) return;
    const span = document.createElement("span");
    span.className = "tok " + probClass(m.p);
    span.textContent = m.s;
    span.addEventListener("click", () => {
      el.genOut.querySelectorAll(".tok.selected").forEach(x => x.classList.remove("selected"));
      span.classList.add("selected");
      el.genInspector.className = "";
      el.genInspector.innerHTML = `
        <div class="ctx-line">It read the last <strong>${m.ctx.toLocaleString()} tokens</strong>${m.cut ? " (its whole context window; older text is forgotten)" : ""} and gave a probability to all ${state.model.cfg.V.toLocaleString()} tokens. The top ones:</div>
        ${barsHtml(m.top, m.s)}
        <p class="small-note">It picked <code>${escapeHtml(showTok(m.s))}</code>, which it gave ${fmtP(m.p)}: ${fmt(m.bits)} bits of information. The entropy of this distribution (how unsure the model was overall) is ${fmt(m.entropy)} bits.</p>`;
    });
    const out = state.genTarget;
    const atBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 30;
    out.appendChild(span);
    if (atBottom) out.scrollTop = out.scrollHeight;
  }

  // ---------------------------------------------------------------------------
  // Step 5: distillation
  // ---------------------------------------------------------------------------
  fillSelect(el.sLayers, LAYERS.slice(0, 4), null, 1);
  fillSelect(el.sWidth, WIDTHS.slice(0, 5), null, 32);
  fillSelect(el.sHeads, HEADS, null, 2);
  syncHeads(el.sWidth, el.sHeads);

  STUDENT_PRESETS.forEach((p, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset" + (i === 0 ? " active" : "");
    b.innerHTML = `<strong>${p.name}</strong><span>${p.note}</span>`;
    b.addEventListener("click", () => {
      el.studentPresets.querySelectorAll(".preset").forEach(x => x.classList.toggle("active", x === b));
      el.sLayers.value = p.L; el.sWidth.value = p.C; el.sHeads.value = p.NH;
      syncHeads(el.sWidth, el.sHeads);
      updateStudentSummary();
    });
    el.studentPresets.appendChild(b);
  });
  [el.sLayers, el.sWidth, el.sHeads].forEach(s => s.addEventListener("change", () => {
    el.studentPresets.querySelectorAll(".preset").forEach(x => x.classList.remove("active"));
    syncHeads(el.sWidth, el.sHeads);
    updateStudentSummary();
  }));

  function studentCfg() { return { L: +el.sLayers.value, C: +el.sWidth.value, NH: +el.sHeads.value }; }
  function distillMode() { return document.querySelector("input[name=d-mode]:checked").value; }
  document.querySelectorAll("input[name=d-mode]").forEach(r => r.addEventListener("change", updateStudentSummary));
  el.dTau.addEventListener("change", updateStudentSummary);

  function updateStudentSummary() {
    if (!state.model) { el.dArch.textContent = ""; return; }
    const t = state.model.cfg, s = studentCfg();
    const n = paramCount({ ...s, V: t.V, T: t.T });
    const mode = distillMode();
    el.dArch.innerHTML = `Student: <strong>${n.toLocaleString()}</strong> weights, ${fmt(100 * n / state.model.params, 0)}% of the teacher's ${state.model.params.toLocaleString()}. ` +
      (mode === "hard" ? "It learns only from the real next token, with no teacher." :
        `For every position, the teacher gives a full probability distribution over ${t.V.toLocaleString()} tokens, and the student learns to match all of it${+el.dTau.value !== 1 ? `. Both are softened with temperature τ = ${el.dTau.value} (and the loss is scaled by τ², as in Hinton's paper)` : ""}.`);
    if (state.distillPaused && distillKey() !== state.distillKey) el.distillBtn.textContent = "Start distilling (new student)";
    else if (state.distillPaused) el.distillBtn.textContent = "Continue distilling";
  }

  function distillKey() { return JSON.stringify([studentCfg(), distillMode(), el.dTau.value]); }

  el.distillBtn.addEventListener("click", async () => {
    const key = distillKey();
    const resume = state.distillPaused && key === state.distillKey;
    if (!resume) { state.dh = newHistory(); state.student = null; }
    state.distillKey = key;
    state.distilling = true;
    refreshButtons();
    const steps = Math.max(1, +el.dSteps.value || 1);
    const B = +el.tBatch.value;
    const mode = distillMode();
    try {
      const res = await call({
        type: "train", distill: true, resume, student: studentCfg(),
        opts: { mode, tau: +el.dTau.value, steps, B, lr: 0.003, warmup: 20, wd: 0.1, evalEvery: Math.max(10, Math.round(steps / 40)), evalRows: Math.max(B, 32) },
      });
      state.student = { ...res.info, val: state.student && state.student.val };
      state.distillPaused = !res.finished;
      el.distillBtn.textContent = res.finished ? "Start distilling (new student)" : "Continue distilling";
      el.distillLabel.textContent = res.finished ? "Finished." : "Paused.";
      if (res.finished) recordRun(mode);
    } catch (err) {
      alert("Distillation stopped: " + err.message);
    }
    state.distilling = false;
    el.distillEta.textContent = "";
    refreshButtons();
  });
  el.dpauseBtn.addEventListener("click", () => state.worker.postMessage({ type: "stop" }));

  function onDistillStart(m) {
    state.student = { ...m.info, val: state.student && state.student.val };
    show(el.distillProgress, true);
    setStudentOptions(true);
  }

  function setStudentOptions(on) {
    [el.genWhich, el.anaWhich].forEach(sel => {
      sel.querySelector("option[value=student]").disabled = !on;
      if (!on) sel.value = "teacher";
    });
  }

  function renderDistillTiles() {
    const h = state.dh, last = h.last;
    const val = h.val.length ? h.val[h.val.length - 1][1] : null;
    const tile = (big, label, sub) => `<div class="stat"><b>${big}</b><span>${label}</span>${sub ? `<span class="sub">${sub}</span>` : ""}</div>`;
    const teach = last && last.ce != null;
    el.distillTiles.innerHTML =
      tile(teach ? fmt(last.ce) : "–", "cross-entropy H(teacher, student)", "bits per token, on this batch") +
      tile(teach ? fmt(last.h) : "–", "teacher's entropy H(teacher)", "the lowest the cross-entropy can go") +
      tile(teach ? fmt(last.kl) : "–", "KL divergence = the difference", "bits wasted per token; 0 = perfect copy") +
      tile(val != null ? fmt(val) : "–", "student's validation loss", h.teacherVal != null ? `teacher: ${fmt(h.teacherVal)} bits per token` : "on the real next tokens");
  }

  function recordRun(mode) {
    const h = state.dh, s = state.student;
    const val = h.val.length ? h.val[h.val.length - 1][1] : null;
    const name = { ce: "Cross-entropy with teacher", kl: "KL divergence from teacher", hard: "No teacher (real tokens only)" }[mode];
    state.runs.push({ name, cfg: `${s.cfg.L} × ${s.cfg.C}`, params: s.params, steps: s.step, tau: mode === "hard" ? "–" : el.dTau.value, val, teacher: h.teacherVal });
    el.distillRuns.innerHTML = `<h3>Your runs</h3>
      <table><tr><th>Loss</th><th>Student (layers × width)</th><th>τ</th><th>Steps</th><th>Student validation loss</th><th>Teacher</th></tr>
      ${state.runs.map(r => `<tr><td>${r.name}</td><td>${r.cfg} (${r.params.toLocaleString()} weights)</td><td>${r.tau}</td><td class="num">${r.steps.toLocaleString()}</td><td class="num"><strong>${fmt(r.val)}</strong></td><td class="num">${fmt(r.teacher)}</td></tr>`).join("")}
      </table>
      <p class="setting-help">Compare a run with cross-entropy and a run with KL divergence (same student, same steps): they train the same way, because the two losses only differ by the teacher's entropy, which the student can't change. Compare both with "No teacher" to see what the teacher's extra information is worth.</p>`;
    show(el.distillRuns, true);
  }

  // ---------------------------------------------------------------------------
  // Step 6: save and load
  // ---------------------------------------------------------------------------
  function toBase64(buf) {
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function fromBase64(str) {
    const s = atob(str), out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out.buffer;
  }

  async function download(which) {
    el.saveStatus.textContent = "Preparing…";
    try {
      const res = await call({ type: "export", which });
      const file = { ...res.model, params: toBase64(res.params) };
      const blob = new Blob([JSON.stringify(file)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `neural-llm-${which}-${res.model.cfg.L}x${res.model.cfg.C}-step${res.model.step}.json`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      el.saveStatus.textContent = `Saved (${fmtBytes(blob.size)}).`;
    } catch (err) {
      el.saveStatus.textContent = "Error: " + err.message;
    }
  }
  el.saveTeacher.addEventListener("click", () => download("teacher"));
  el.saveStudent.addEventListener("click", () => download("student"));

  el.loadModel.addEventListener("change", async () => {
    const f = el.loadModel.files[0];
    el.loadModel.value = "";
    if (!f) return;
    if ((state.training || state.distilling) && !confirm("Stop training and load this model?")) return;
    el.saveStatus.textContent = "Loading…";
    try {
      const file = JSON.parse(await f.text());
      const params = fromBase64(file.params || "");
      delete file.params;
      const res = await call({ type: "import", model: file, params }, [params]);
      resetModel();
      state.model = res.info;
      state.hasData = false;
      state.tok = null;
      const c = res.info.cfg;
      el.aLayers.value = c.L; el.aWidth.value = c.C; el.aHeads.value = c.NH; el.aCtx.value = c.T;
      el.tokKind.value = res.tokenizer.kind;
      show(el.vocabField, res.tokenizer.kind === "bpe");
      el.reuse.checked = true;
      show(el.tokStats, false);
      el.createStats.innerHTML = `<p class="note">Loaded a model with <strong>${res.info.params.toLocaleString()}</strong> weights, trained for ${res.info.step.toLocaleString()} steps (${res.tokenizer.kind === "bpe" ? `${res.tokenizer.vocab.toLocaleString()}-token BPE` : "character"} tokenizer). You can generate text now. To train it more, add text in step 1 and keep "Keep the current model" ticked.</p>`;
      show(el.createStats, true);
      updateArchSummary();
      el.saveStatus.textContent = "Loaded.";
    } catch (err) {
      el.saveStatus.textContent = "Could not load: " + err.message;
    }
    refreshButtons();
  });

  // ---------------------------------------------------------------------------
  // Buttons
  // ---------------------------------------------------------------------------
  function refreshButtons() {
    const busy = state.tokenizing || state.training || state.distilling;
    const haveText = state.source === "upload" ? state.files.length > 0 : state.chars > 0;
    el.tokBtn.disabled = busy || !haveText;
    show(el.reuseWrap, !!state.model);
    enableCard(el.archCard, !!state.tok);
    el.createBtn.disabled = busy || !state.tok || !state.hasData;
    enableCard(el.trainCard, !!state.model);
    el.trainBtn.disabled = busy || !state.model || !state.hasData;
    show(el.pauseBtn, state.training);
    el.anaBtn.disabled = !state.model || (!state.hasData && !el.anaInput.value.trim());
    enableCard(el.genCard, !!state.model);
    el.genInput.disabled = !state.model;
    el.genBtn.disabled = !state.model || state.generating;
    show(el.genStop, state.generating);
    enableCard(el.distillCard, !!state.model && state.hasData);
    el.distillBtn.disabled = busy || !state.model || !state.hasData;
    show(el.dpauseBtn, state.distilling);
    enableCard(el.saveCard, true);
    el.saveTeacher.disabled = !state.model;
    el.saveStudent.disabled = !state.student;
    updateStudentSummary();
  }
  el.anaInput.addEventListener("input", refreshButtons);

  // ---------------------------------------------------------------------------
  // Lesson widgets
  // ---------------------------------------------------------------------------
  const log2 = Math.log2;
  const SVG_NS = "http://www.w3.org/2000/svg";

  // Cross-entropy diagram: bar widths are p, heights are -log2 q.
  const CE_PRESETS = [
    { name: "Robot, part 1: the code fits", labels: ["↑ up", "↓ down", "← left", "→ right"], p: [0.5, 0.25, 0.125, 0.125], q: [0.5, 0.25, 0.125, 0.125] },
    { name: "Robot, part 2: the plan changes", labels: ["↑ up", "↓ down", "← left", "→ right"], p: [0.125, 0.125, 0.25, 0.5], q: [0.5, 0.25, 0.125, 0.125] },
    { name: "Even code, skewed world", labels: ["A", "B"], p: [0.9, 0.1], q: [0.5, 0.5] },
    { name: "Skewed code, even world", labels: ["A", "B"], p: [0.5, 0.5], q: [0.9, 0.1] },
  ];
  const ce = { ...structuredClone(CE_PRESETS[1]), preset: 1 };

  function setProb(arr, i, v) {
    v = Math.min(0.99, Math.max(0.01, v));
    const rest = arr.reduce((a, x, j) => j === i ? a : a + x, 0);
    const left = 1 - v;
    for (let j = 0; j < arr.length; j++) if (j !== i) arr[j] = rest > 0 ? arr[j] * left / rest : left / (arr.length - 1);
    arr[i] = v;
  }

  // Shannon code: lengths ceil(-log2 q), assigned in order (canonical code).
  function shannonCode(q) {
    const lens = q.map(x => Math.max(1, Math.ceil(-log2(x) - 1e-9)));
    if (Math.max(...lens) > 8) return null;
    const order = lens.map((l, i) => [l, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const codes = [];
    let code = 0, prev = order[0][0];
    order.forEach(([l, i], k) => {
      if (k) { code = (code + 1) << (l - prev); prev = l; }
      codes[i] = code.toString(2).padStart(l, "0");
    });
    return codes;
  }

  function renderCE() {
    const root = $("ce-widget");
    const H = ce.p.reduce((a, p) => a - p * log2(p), 0);
    const X = ce.p.reduce((a, p, i) => a - p * log2(ce.q[i]), 0);
    const codes = shannonCode(ce.q);
    root.innerHTML = `
      <div class="presets">${CE_PRESETS.map((p, i) => `<button type="button" class="preset${i === ce.preset ? " active" : ""}" data-i="${i}"><strong>${p.name}</strong></button>`).join("")}</div>
      <div class="widget-row" style="margin-top:12px">
        <div><svg id="ce-svg" viewBox="0 0 440 270" role="img" aria-label="Cross-entropy diagram"></svg></div>
        <div>
          <div class="ce-table-wrap"><table class="ce-table">
            <tr><th>Symbol</th><th class="p-col">Real world p</th><th>Code made for q</th><th>bits = −log₂ q</th>${codes ? "<th>code</th>" : ""}</tr>
            ${ce.labels.map((l, i) => `<tr>
              <td>${l}</td>
              <td class="p-col"><input type="range" min="1" max="99" value="${Math.round(ce.p[i] * 100)}" data-k="p" data-i="${i}" aria-label="p of ${l}"><span class="muted">${fmtP(ce.p[i])}</span></td>
              <td><input type="range" min="1" max="99" value="${Math.round(ce.q[i] * 100)}" data-k="q" data-i="${i}" aria-label="q of ${l}"><span class="muted">${fmtP(ce.q[i])}</span></td>
              <td class="num">${fmt(-log2(ce.q[i]), 3)}</td>
              ${codes ? `<td><code>${codes[i]}</code></td>` : ""}</tr>`).join("")}
          </table></div>
          <div class="ce-results">
            <div class="stat"><b>${fmt(X, 3)}</b><span>cross-entropy H(p, q): average bits with this code</span></div>
            <div class="stat"><b>${fmt(H, 3)}</b><span>entropy H(p): the best possible</span></div>
            <div class="stat"><b>${fmt(X - H, 3)}</b><span>KL divergence: bits wasted</span></div>
          </div>
        </div>
      </div>`;
    drawCE($("ce-svg"));
    root.querySelectorAll(".preset").forEach(b => b.addEventListener("click", () => {
      Object.assign(ce, structuredClone(CE_PRESETS[+b.dataset.i]), { preset: +b.dataset.i });
      renderCE();
    }));
    root.querySelectorAll("input[type=range]").forEach(r => r.addEventListener("change", () => {
      setProb(ce[r.dataset.k], +r.dataset.i, +r.value / 100);
      ce.preset = -1;
      renderCE();
    }));
  }

  function svgEl(tag, attrs, text) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  function drawCE(svg) {
    const L = 46, R = 425, T = 14, B = 230;
    const maxBits = Math.max(4, Math.ceil(Math.max(...ce.q.map(q => -log2(q)), ...ce.p.map(p => -log2(p)))));
    const Y = b => B - (B - T) * b / maxBits, Xp = x => L + (R - L) * x;
    const accent = cssVar("--accent"), soft = cssVar("--bar"), muted = cssVar("--muted"), orange = cssVar("--line-val");
    for (let b = 0; b <= maxBits; b++) {
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: Y(b), y2: Y(b), stroke: cssVar("--border") }));
      svg.appendChild(svgEl("text", { x: L - 6, y: Y(b) + 4, "text-anchor": "end", class: "svg-muted" }, b));
    }
    svg.appendChild(svgEl("text", { x: 12, y: (T + B) / 2, transform: `rotate(-90 12 ${(T + B) / 2})`, "text-anchor": "middle", class: "svg-muted" }, "height: −log₂ q (bits)"));
    svg.appendChild(svgEl("text", { x: (L + R) / 2, y: 266, "text-anchor": "middle", class: "svg-muted" }, "width: p (how often it really happens)"));
    let x = 0;
    ce.p.forEach((p, i) => {
      const hq = -log2(ce.q[i]), hp = -log2(p);
      svg.appendChild(svgEl("rect", { x: Xp(x) + 1, y: Y(hq), width: Math.max(1, Xp(x + p) - Xp(x) - 2), height: B - Y(hq), fill: soft, stroke: accent, "stroke-width": 1.5 }));
      svg.appendChild(svgEl("rect", { x: Xp(x) + 1, y: Y(hp), width: Math.max(1, Xp(x + p) - Xp(x) - 2), height: B - Y(hp), fill: "none", stroke: orange, "stroke-width": 1.5, "stroke-dasharray": "5 3" }));
      const cx = (Xp(x) + Xp(x + p)) / 2;
      if (Xp(x + p) - Xp(x) > 26) {
        svg.appendChild(svgEl("text", { x: cx, y: B + 16, "text-anchor": "middle", class: "svg-text" }, ce.labels[i].split(" ")[0]));
        svg.appendChild(svgEl("text", { x: cx, y: Math.min(Y(hq), Y(hp)) - 4, "text-anchor": "middle", class: "svg-muted" }, fmt(hq, 2)));
      }
      x += p;
    });
    svg.appendChild(svgEl("line", { x1: L, x2: R, y1: B, y2: B, stroke: muted }));
  }

  // Two outcomes: cross-entropy as a function of q1, for a fixed p1.
  const two = { p: 0.3, q: 0.7 };
  function renderTwo() {
    const root = $("two-widget");
    const H2 = x => (x <= 0 || x >= 1) ? 0 : -x * log2(x) - (1 - x) * log2(1 - x);
    const CE2 = (p, q) => -p * log2(q) - (1 - p) * log2(1 - q);
    root.innerHTML = `
      <div class="widget-row">
        <div><svg id="two-svg" viewBox="0 0 440 280" role="img" aria-label="Cross-entropy for two outcomes"></svg></div>
        <div>
          <div class="setting"><label for="two-p">Real world: p₁ <span class="val">${fmt(two.p)}</span></label><input type="range" id="two-p" min="1" max="99" value="${Math.round(two.p * 100)}"></div>
          <div class="setting"><label for="two-q">Code made for: q₁ <span class="val">${fmt(two.q)}</span></label><input type="range" id="two-q" min="1" max="99" value="${Math.round(two.q * 100)}"></div>
          <div class="ce-results">
            <div class="stat"><b>${fmt(CE2(two.p, two.q), 3)}</b><span>H(p, q) at this q</span></div>
            <div class="stat"><b>${fmt(H2(two.p), 3)}</b><span>minimum, at q = p: the entropy H(p)</span></div>
          </div>
          <p class="setting-help">Fix p, slide q: the purple dot is lowest when q₁ = p₁. Now slide p: the whole curve moves, and its lowest point always sits on the green entropy curve.</p>
        </div>
      </div>`;
    const svg = $("two-svg");
    const L = 40, R = 425, T = 12, B = 240, maxY = 3.5;
    const X = x => L + (R - L) * x, Y = y => B - (B - T) * Math.min(y, maxY) / maxY;
    const accent = cssVar("--accent"), green = cssVar("--green"), muted = cssVar("--muted"), border = cssVar("--border");
    for (let b = 0; b <= 3; b++) {
      svg.appendChild(svgEl("line", { x1: L, x2: R, y1: Y(b), y2: Y(b), stroke: border }));
      svg.appendChild(svgEl("text", { x: L - 6, y: Y(b) + 4, "text-anchor": "end", class: "svg-muted" }, b));
    }
    for (const t of [0, 0.25, 0.5, 0.75, 1]) svg.appendChild(svgEl("text", { x: X(t), y: B + 16, "text-anchor": "middle", class: "svg-muted" }, t));
    svg.appendChild(svgEl("text", { x: (L + R) / 2, y: 274, "text-anchor": "middle", class: "svg-muted" }, "q₁ (and, for the green curve, p₁)"));
    const path = (f) => {
      let d = "";
      for (let i = 1; i < 200; i++) { const x = i / 200; d += (d ? "L" : "M") + X(x).toFixed(1) + "," + Y(f(x)).toFixed(1); }
      return d;
    };
    svg.appendChild(svgEl("path", { d: path(H2), fill: "none", stroke: green, "stroke-width": 2.5 }));
    svg.appendChild(svgEl("path", { d: path(q => CE2(two.p, q)), fill: "none", stroke: accent, "stroke-width": 2.5 }));
    svg.appendChild(svgEl("line", { x1: X(two.p), x2: X(two.p), y1: T, y2: B, stroke: muted, "stroke-dasharray": "4 3" }));
    svg.appendChild(svgEl("circle", { cx: X(two.p), cy: Y(H2(two.p)), r: 5, fill: green }));
    svg.appendChild(svgEl("circle", { cx: X(two.q), cy: Y(CE2(two.p, two.q)), r: 6, fill: accent }));
    svg.appendChild(svgEl("line", { x1: L, x2: R, y1: B, y2: B, stroke: muted }));
    $("two-p").addEventListener("input", e => { two.p = +e.target.value / 100; renderTwo(); $("two-p").focus(); });
    $("two-q").addEventListener("input", e => { two.q = +e.target.value / 100; renderTwo(); $("two-q").focus(); });
  }

  // Language trees and zipping: how many extra bits does a snippet of B cost
  // after gzip has read A?
  async function gz(str) {
    const stream = new Blob([str]).stream().pipeThrough(new CompressionStream("gzip"));
    return (await new Response(stream).arrayBuffer()).byteLength;
  }

  function renderZip() {
    const root = $("zip-widget");
    if (typeof CompressionStream === "undefined") {
      root.innerHTML = `<p class="muted">Your browser has no built-in gzip (CompressionStream), so this demo can't run.</p>`;
      return;
    }
    root.innerHTML = `
      <p class="setting-help">For every pair of texts: cut a snippet b (the last 500 characters) off text B. Then measure how many extra bits per character b costs when gzip has just read text A. The diagonal is b after the rest of its own text. A small number means "A and B have similar patterns". Pasted text in step 1 joins the comparison if it has at least 2,000 characters.</p>
      <button class="btn ghost small" id="zip-run" type="button">Run the zipping experiment</button>
      <div class="zip-out" id="zip-out"></div>`;
    $("zip-run").addEventListener("click", runZip);
  }

  async function runZip() {
    const out = $("zip-out");
    const docs = samples.map(s => ({ name: s.title, text: s.text }));
    if (el.text.value.length >= 2000) docs.push({ name: "Your pasted text", text: el.text.value.slice(0, 30000) });
    out.textContent = "Compressing…";
    const SNIP = 500;
    const parts = docs.map(d => ({ rest: d.text.slice(0, -SNIP), snip: d.text.slice(-SNIP) }));
    const base = await Promise.all(parts.map(p => gz(p.rest)));
    const m = [];
    for (let a = 0; a < docs.length; a++) {
      m.push([]);
      for (let b = 0; b < docs.length; b++) {
        const extra = (await gz(parts[a].rest + parts[b].snip)) - base[a];
        m[a].push(extra * 8 / parts[b].snip.length);
      }
    }
    const min = Math.min(...m.flat()), max = Math.max(...m.flat());
    const shade = v => `background: color-mix(in srgb, var(--accent) ${Math.round(60 * (1 - (v - min) / Math.max(1e-9, max - min)))}%, transparent)`;
    out.innerHTML = `
      <table><tr><th>gzip read ↓ · then snippet of →</th>${docs.map(d => `<th>${escapeHtml(d.name)}</th>`).join("")}</tr>
      ${docs.map((d, a) => `<tr><th>${escapeHtml(d.name)}</th>${m[a].map(v => `<td style="${shade(v)}">${fmt(v)}</td>`).join("")}</tr>`).join("")}
      </table>
      <p class="setting-help">Extra bits per character for the snippet. Each column's darkest cell shows which text "prepares" gzip best for that snippet, usually its own text. The authors of the paper turned numbers like these into distances between languages and rebuilt the family tree of European languages from them.</p>`;
  }

  // ---------------------------------------------------------------------------
  // Start
  // ---------------------------------------------------------------------------
  makeWorker();
  updateCounter();
  updateArchSummary();
  renderTrainTiles();
  renderDistillTiles();
  refreshButtons();
  renderCE();
  renderTwo();
  renderZip();
  drawSoon();
})();
