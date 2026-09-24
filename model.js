// The language model. This function runs inside a Web Worker. app.js turns its
// source into a Blob, so the site works from GitHub Pages and from file:// alike.
// It must not use anything from outside its own body.
function workerMain() {
  "use strict";

  const PARA = "\u2029";                // paragraph break token
  const TOKEN_RE = /\u2029|[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*|\.{2,}|[^\s\p{L}\p{N}]/gu;
  const WORDY = /[\p{L}\p{N}]/u;
  const END_TOKENS = new Set([".", "!", "?", "…", "...", PARA]);
  const MAX_CTX = 1024;                 // longest context the index is sorted for
  const RANGE_CAP = 150000;             // sample at most this many matches when counting
  const CHUNK = 1 << 20;                // characters fed to the tokenizer at a time

  // ----- the model -----
  let N = 0;              // number of tokens
  let surf = null;        // Int32Array: surface-form id at each position
  let s2k = null;         // Int32Array: surface id -> key id (lower-cased form, used for matching)
  let sa = null;          // Int32Array: suffix array (every position, sorted by the tokens that follow it)
  let depth = 1;          // sa is sorted by at least this many tokens
  let surfList = [];      // surface id -> string
  let keyList = [];       // key id -> string
  let keyIndex = new Map();
  let keyCount = null;    // Int32Array: frequency of each key
  let unigram = null;     // [{id, c}] surface ids by frequency

  const key = (p) => s2k[surf[p]];
  function post(msg) { self.postMessage(msg); }

  // ---------------------------------------------------------------------------
  // Reading text
  // ---------------------------------------------------------------------------
  function prep(text, keepLines) {
    text = text.replace(/\r\n?/g, "\n");
    if (keepLines) return text.replace(/\n+/g, PARA);
    return text.replace(/[ \t]*\n[ \t]*\n\s*/g, PARA).replace(/\n/g, " ");
  }

  function tokenize(text) {
    return text.match(TOKEN_RE) || [];
  }

  function isWs(c) {
    return c <= 32 || c === 160 || c === 0x2028 || c === 0x2029 || c === 0x3000 || c === 0xfeff;
  }

  // Streams text into token ids without ever holding a list of token strings.
  function makeReader(keepLines, maxWords) {
    let buf = new Int32Array(1 << 20);
    let n = 0, words = 0, last = -1;
    let carry = "";
    const surfIndex = new Map();
    const isWord = [];
    const s2kArr = [];
    surfList = []; keyList = []; keyIndex = new Map();
    let paraId = -1;

    function idOf(t) {
      let s = surfIndex.get(t);
      if (s === undefined) {
        s = surfList.length;
        surfList.push(t);
        surfIndex.set(t, s);
        isWord.push(WORDY.test(t));
        const k = t.toLowerCase();
        let kid = keyIndex.get(k);
        if (kid === undefined) { kid = keyList.length; keyList.push(k); keyIndex.set(k, kid); }
        s2kArr.push(kid);
        if (t === PARA) paraId = s;
      }
      return s;
    }

    function feed(str) {
      TOKEN_RE.lastIndex = 0;
      let m;
      while ((m = TOKEN_RE.exec(str))) {
        const s = idOf(m[0]);
        if (s === paraId && (last === -1 || last === paraId)) continue;
        if (isWord[s]) { if (words >= maxWords) return false; words++; }
        if (n === buf.length) { const nb = new Int32Array(buf.length * 2); nb.set(buf); buf = nb; }
        buf[n++] = s;
        last = s;
      }
      return true;
    }

    return {
      // Returns false once the word limit is reached.
      push(text, final) {
        let b = carry + text;
        let cut = b.length;
        if (!final) {
          // Stop before the last run of whitespace, so no word or "\n\n" is split in two.
          let i = b.length - 1;
          while (i >= 0 && !isWs(b.charCodeAt(i))) i--;
          while (i >= 0 && isWs(b.charCodeAt(i))) i--;
          cut = i + 1;
          if (cut <= 0) { carry = b; return true; }
        }
        carry = b.slice(cut);
        return feed(prep(cut === b.length ? b : b.slice(0, cut), keepLines));
      },
      separate() { // paragraph break between files
        if (carry) { this.push("", true); carry = ""; }
        feed(PARA);
      },
      finish() {
        if (carry) this.push("", true);
        return { ids: buf.slice(0, n), words, s2k: Int32Array.from(s2kArr) };
      },
      get words() { return words; },
    };
  }

  // ---------------------------------------------------------------------------
  // Suffix array by prefix doubling with radix sort. After the round that uses
  // step k, positions are sorted by their first 2k tokens. Stops when every
  // position is unique or MAX_CTX tokens are sorted.
  // ---------------------------------------------------------------------------
  function buildSuffixArray() {
    const n = N, K = keyList.length;
    let rank = new Int32Array(n);
    for (let i = 0; i < n; i++) rank[i] = key(i);
    let out = new Int32Array(n);
    let aux = new Int32Array(n);
    const cnt = new Int32Array(Math.max(n, K) + 1);

    // sort by first token
    for (let i = 0; i < n; i++) cnt[rank[i]]++;
    for (let i = 0, s = 0; i < K; i++) { const c = cnt[i]; cnt[i] = s; s += c; }
    for (let i = 0; i < n; i++) out[cnt[rank[i]]++] = i;

    let classes = K;
    let d = 1;
    while (classes < n && d < MAX_CTX) {
      const k = d;
      post({ type: "progress", phase: `Sorting by the next ${2 * k} tokens (${Math.round(100 * classes / n)}% of positions are unique so far)…`, pct: 45 + 50 * Math.log2(2 * k) / Math.log2(MAX_CTX) });
      // order by second key: positions whose second half runs off the end come first
      let p = 0;
      for (let i = Math.max(0, n - k); i < n; i++) aux[p++] = i;
      for (let i = 0; i < n; i++) { const j = out[i]; if (j >= k) aux[p++] = j - k; }
      // stable counting sort by first key
      cnt.fill(0, 0, classes + 1);
      for (let i = 0; i < n; i++) cnt[rank[i]]++;
      for (let i = 0, s = 0; i < classes; i++) { const c = cnt[i]; cnt[i] = s; s += c; }
      for (let i = 0; i < n; i++) { const j = aux[i]; out[cnt[rank[j]]++] = j; }
      // new ranks
      const nr = aux;
      let c = 0;
      nr[out[0]] = 0;
      for (let i = 1; i < n; i++) {
        const a = out[i - 1], b = out[i];
        if (rank[a] !== rank[b] || (a + k < n ? rank[a + k] : -1) !== (b + k < n ? rank[b + k] : -1)) c++;
        nr[b] = c;
      }
      classes = c + 1;
      aux = rank;
      rank = nr;
      d = 2 * k;
    }
    sa = out;
    depth = classes >= n ? Infinity : d;
  }

  // ---------------------------------------------------------------------------
  // Building
  // ---------------------------------------------------------------------------
  async function build(m) {
    const t0 = performance.now();
    sa = surf = null; // free the old model first
    const reader = makeReader(m.keepLines, m.maxWords);
    let ok = true;

    if (m.files && m.files.length) {
      const total = m.files.reduce((a, f) => a + f.size, 0) || 1;
      let done = 0;
      for (let fi = 0; fi < m.files.length && ok; fi++) {
        if (fi > 0) reader.separate();
        const r = m.files[fi].stream().pipeThrough(new TextDecoderStream()).getReader();
        let fileDone = 0;
        for (;;) {
          const { value, done: end } = await r.read();
          if (end) break;
          fileDone += value.length;
          ok = reader.push(value, false);
          post({ type: "progress", phase: `Reading ${m.files[fi].name}… ${reader.words.toLocaleString()} words so far`, pct: 40 * Math.min(1, (done + fileDone) / total) });
          if (!ok) { r.cancel(); break; }
        }
        done += m.files[fi].size;
      }
    } else {
      const text = m.text || "";
      for (let i = 0; i < text.length && ok; i += CHUNK) {
        ok = reader.push(text.slice(i, i + CHUNK), false);
        post({ type: "progress", phase: `Reading your text… ${reader.words.toLocaleString()} words so far`, pct: 40 * Math.min(1, i / text.length) });
      }
    }

    const res = reader.finish();
    surf = res.ids;
    s2k = res.s2k;
    N = surf.length;
    if (N === 0) throw new Error("No words found in that text.");

    post({ type: "progress", phase: "Counting tokens…", pct: 42 });
    keyCount = new Int32Array(keyList.length);
    const surfCount = new Int32Array(surfList.length);
    for (let i = 0; i < N; i++) { surfCount[surf[i]]++; keyCount[key(i)]++; }
    unigram = [];
    for (let s = 0; s < surfCount.length; s++) unigram.push({ id: s, c: surfCount[s] });
    unigram.sort((a, b) => b.c - a.c);
    if (unigram.length > 5000) unigram.length = 5000;

    buildSuffixArray();

    post({ type: "progress", phase: "Collecting stats…", pct: 97 });
    const topWords = [];
    const byFreq = Array.from(keyCount.keys()).sort((a, b) => keyCount[b] - keyCount[a]);
    for (const kid of byFreq) {
      if (WORDY.test(keyList[kid])) topWords.push([keyList[kid], keyCount[kid]]);
      if (topWords.length >= 15) break;
    }
    let pairs = 0;
    for (let i = 1; i < N; i++) {
      const a = sa[i - 1], b = sa[i];
      if (key(a) !== key(b) || a + 1 >= N || b + 1 >= N || key(a + 1) !== key(b + 1)) pairs++;
    }

    post({
      type: "built",
      stats: {
        tokens: N,
        words: res.words,
        truncated: !ok,
        vocab: keyList.length,
        pairs,
        depth: depth === Infinity ? null : depth,
        topWords,
        ms: Math.round(performance.now() - t0),
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Looking things up
  // ---------------------------------------------------------------------------
  // Compare the tokens at pos with ctx. Running off the end sorts first,
  // matching the order of the suffix array.
  function cmp(pos, ctx) {
    for (let j = 0; j < ctx.length; j++) {
      const p = pos + j;
      if (p >= N) return -1;
      const d = key(p) - ctx[j];
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

  function suffix(history, k) { return history.slice(history.length - k); }

  // Number of different tokens seen after ctx, stopping once `need` is reached.
  function variety(ctx, need) {
    if (ctx.some(x => x < 0)) return 0;
    const [lo, hi] = range(ctx);
    const k = ctx.length;
    let first = -1;
    for (let i = lo; i < hi; i++) {
      const p = sa[i] + k;
      if (p >= N) continue;
      if (first === -1) { first = surf[p]; if (need <= 1) return 1; }
      else if (surf[p] !== first) return 2;
    }
    return first === -1 ? 0 : 1;
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
    return { items, total, sampled: step > 1 };
  }

  // Longest k (up to maxK) where ok(k) holds. ok must be true for small k and
  // false for large k, so a binary search works.
  function longest(maxK, ok) {
    let lo = 0, hi = maxK;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ok(mid)) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  // Use the longest stretch of the history that the text has seen before
  // ("backoff"). With avoidCopy, stop at the longest stretch that still had
  // at least two different next tokens.
  function distribution(history, order, avoidCopy) {
    const maxK = Math.min(order, history.length, depth, MAX_CTX);
    let k = longest(maxK, kk => variety(suffix(history, kk), 1) >= 1);
    if (avoidCopy && k > 1) {
      const k2 = longest(k, kk => kk <= 1 || variety(suffix(history, kk), 2) >= 2);
      k = Math.max(1, k2);
    }
    let ctx = k > 0 ? suffix(history, k) : [];
    let res = ctx.length ? nextCounts(ctx) : null;
    if (!res) { ctx = []; res = nextCounts([]); }
    return { k: ctx.length, ctx, ...res };
  }

  function applyTemperature(items, temp) {
    if (temp <= 0.001) {
      const best = items[0].c;
      const ties = items.filter(x => x.c === best).length;
      return items.map(x => (x.c === best ? 1 / ties : 0));
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

  const SHOW_CTX = 40;
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
      ctx: dist.ctx.slice(-SHOW_CTX).map(id => keyList[id]),
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

  function trimmed(items) { return items.length > 2000 ? items.slice(0, 2000) : items; }

  function generate(prompt, s) {
    const pk = promptKeys(prompt);
    const steps = [];
    let history, how;

    // 1) Has the text seen the end of the message? Then just continue it.
    const best = longest(Math.min(s.order, pk.length, depth, MAX_CTX), kk => variety(suffix(pk, kk), 1) >= 1);
    if (best >= 2) {
      history = pk.slice();
      how = { mode: "continue", words: best };
    } else {
      // 2) Otherwise start from the rarest word in the message that the model knows.
      let pick = -1;
      for (const k of pk) {
        if (k >= 0 && WORDY.test(keyList[k]) && (pick < 0 || keyCount[k] < keyCount[pick])) pick = k;
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

    const maxTokens = [0, 30, 70, 160, 400][s.length];
    const maxSentences = [0, 1, 3, 8, 25][s.length];
    let sentences = 0, produced = steps.length;
    for (let guard = 0; guard < maxTokens * 3 && produced < maxTokens; guard++) {
      const dist = distribution(history, s.order, s.avoidCopy);
      if (!dist.items || dist.items.length === 0) break;
      dist.items = trimmed(dist.items);
      const probs = applyTemperature(dist.items, s.temp);
      const idx = sample(probs);
      const tok = surfList[dist.items[idx].id];
      history.push(s2k[dist.items[idx].id]);
      if (history.length > 2 * MAX_CTX) history = history.slice(-MAX_CTX);
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
    const items = trimmed(dist.items);
    const info = describe({ ...dist, items }, applyTemperature(items, s.temp), 0);
    info.chosen = null;
    info.unknown = tokenize(prep(text, false)).filter((t, i) => pk[i] < 0);
    return info;
  }

  self.onmessage = async (e) => {
    const m = e.data;
    try {
      if (m.type === "build") await build(m);
      else if (!sa) throw new Error("The model is not built yet.");
      else if (m.type === "generate") post({ type: "generated", id: m.id, ...generate(m.prompt, m.settings) });
      else if (m.type === "predict") post({ type: "prediction", id: m.id, info: predict(m.text, m.settings) });
    } catch (err) {
      const oom = err instanceof RangeError || /memory|allocation/i.test(String(err && err.message));
      post({
        type: "error", id: m.id,
        message: oom
          ? "Your browser ran out of memory. Try a smaller text, or close other tabs and try again."
          : String(err && err.message || err),
      });
    }
  };
}
