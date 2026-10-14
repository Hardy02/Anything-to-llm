// A small GPT-style transformer that is trained from scratch, in the browser.
// This function runs inside a Web Worker. train.js turns its source into a Blob,
// so the page works from GitHub Pages and from file:// alike. It must not use
// anything from outside its own body.
//
// The maths follows Andrej Karpathy's llm.c (train_gpt2.c): every layer has a
// forward pass and a hand-written backward pass (backpropagation). There is no
// machine-learning library.
function nnWorkerMain() {
  "use strict";

  // Tests can set self.NN_FLOAT = Float64Array to check gradients exactly.
  const F = self.NN_FLOAT || Float32Array;
  const LN2 = Math.LN2;
  const MAX_VOCAB = 8192;              // token ids must fit in a Uint16Array
  const MAX_BASE_CHARS = 1024;         // characters kept as single tokens
  const BPE_SAMPLE = 1 << 21;          // BPE merges are learned from the first 2M characters
  const UNK = "�";                // token for characters that are not in the vocabulary

  function post(msg, transfer) { self.postMessage(msg, transfer || []); }

  // ---------------------------------------------------------------------------
  // Random numbers (seeded, so a run can be repeated)
  // ---------------------------------------------------------------------------
  let seed = 1337;
  function rand() {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function randn() {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // Lets "stop" messages arrive between training steps, without the 4 ms
  // minimum delay of a nested setTimeout.
  const yieldChannel = new MessageChannel();
  const yieldQueue = [];
  yieldChannel.port1.onmessage = () => { const r = yieldQueue.shift(); if (r) r(); };
  function yieldNow() { return new Promise(r => { yieldQueue.push(r); yieldChannel.port2.postMessage(0); }); }

  // ---------------------------------------------------------------------------
  // Tokenizer: single characters, or byte-pair encoding (BPE) subwords
  // ---------------------------------------------------------------------------
  // Like GPT-2, text is first cut into words (with their leading space),
  // numbers, punctuation and whitespace. BPE merges never cross these cuts.
  const PRE_RE = / ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+/gu;

  function countChars(text) {
    const counts = new Map();
    for (const ch of text) counts.set(ch, (counts.get(ch) || 0) + 1);
    return counts;
  }

  function baseVocab(text) {
    const counts = [...countChars(text)].sort((a, b) => b[1] - a[1]);
    const chars = counts.slice(0, MAX_BASE_CHARS).map(x => x[0]);
    const dropped = counts.length - chars.length;
    // Characters in frequency order, so the most common get the smallest ids.
    return { vocab: [UNK, ...chars.filter(c => c !== UNK)], dropped };
  }

  function makeTokenizer(kind, vocab, merges) {
    const charId = new Map();
    for (let i = 0; i < vocab.length; i++) if ([...vocab[i]].length === 1 && !charId.has(vocab[i])) charId.set(vocab[i], i);
    // A merge of (a, b) makes token id baseCount + rank. A lower rank was learned earlier, so it is applied first.
    const baseCount = vocab.length - merges.length;
    const rank = new Map();
    merges.forEach(([a, b], r) => rank.set(a * MAX_VOCAB + b, r));
    const cache = new Map();

    function charIds(str) {
      const out = [];
      for (const ch of str) { const id = charId.get(ch); out.push(id === undefined ? 0 : id); }
      return out;
    }

    function bpe(word) {
      let ids = cache.get(word);
      if (ids) return ids;
      ids = charIds(word);
      while (ids.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < ids.length - 1; i++) {
          const r = rank.get(ids[i] * MAX_VOCAB + ids[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const a = ids[best], b = ids[best + 1], id = baseCount + bestRank;
        const next = [];
        for (let i = 0; i < ids.length; i++) {
          if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) { next.push(id); i++; }
          else next.push(ids[i]);
        }
        ids = next;
      }
      if (cache.size < 200000) cache.set(word, ids);
      return ids;
    }

    // Returns a Uint16Array of token ids.
    function encode(text, onProgress) {
      const out = new Uint16Array(Math.max(1, text.length));
      let n = 0;
      if (kind === "char") {
        for (const ch of text) { const id = charId.get(ch); out[n++] = id === undefined ? 0 : id; }
        return out.slice(0, n);
      }
      PRE_RE.lastIndex = 0;
      let m, count = 0;
      while ((m = PRE_RE.exec(text))) {
        const ids = bpe(m[0]);
        for (let i = 0; i < ids.length; i++) out[n++] = ids[i];
        if (onProgress && (++count & 0xffff) === 0) onProgress(m.index / text.length);
      }
      return out.slice(0, n);
    }

    function decode(ids) {
      let s = "";
      for (let i = 0; i < ids.length; i++) s += vocab[ids[i]];
      return s;
    }

    return { kind, vocab, merges, encode, decode, size: vocab.length };
  }

  // Learns BPE merges: start from single characters, then again and again join
  // the pair of neighbouring tokens that appears most often.
  function trainBPE(text, vocabSize, progress) {
    const { vocab, dropped } = baseVocab(text);
    const charId = new Map(vocab.map((c, i) => [c, i]));
    const sample = text.length > BPE_SAMPLE ? text.slice(0, BPE_SAMPLE) : text;
    const wordCounts = new Map();
    PRE_RE.lastIndex = 0;
    let m;
    while ((m = PRE_RE.exec(sample))) wordCounts.set(m[0], (wordCounts.get(m[0]) || 0) + 1);

    const words = [], freq = [];
    for (const [w, c] of wordCounts) {
      const ids = [];
      for (const ch of w) { const id = charId.get(ch); ids.push(id === undefined ? 0 : id); }
      if (ids.length > 1) { words.push(ids); freq.push(c); }
    }
    const pairCount = new Map();   // pair key -> how often it appears
    const where = new Map();       // pair key -> Set of word indices that (may) contain it
    function addPairs(wi, sign) {
      const w = words[wi], f = freq[wi] * sign;
      for (let i = 0; i < w.length - 1; i++) {
        const k = w[i] * MAX_VOCAB + w[i + 1];
        const c = (pairCount.get(k) || 0) + f;
        if (c > 0) pairCount.set(k, c); else pairCount.delete(k);
        if (sign > 0) { let s = where.get(k); if (!s) where.set(k, s = new Set()); s.add(wi); }
      }
    }
    for (let i = 0; i < words.length; i++) addPairs(i, 1);

    const merges = [];
    const target = Math.min(MAX_VOCAB, vocabSize);
    while (vocab.length < target) {
      let bestK = -1, bestC = 1;
      for (const [k, c] of pairCount) if (c > bestC || (c === bestC && bestK >= 0 && k < bestK)) { bestC = c; bestK = k; }
      if (bestK < 0) break;              // no pair appears twice: nothing left to learn
      const a = Math.floor(bestK / MAX_VOCAB), b = bestK % MAX_VOCAB, id = vocab.length;
      vocab.push(vocab[a] + vocab[b]);
      merges.push([a, b]);
      const list = where.get(bestK);
      where.delete(bestK);
      for (const wi of list) {
        const w = words[wi];
        let has = false;
        for (let i = 0; i < w.length - 1; i++) if (w[i] === a && w[i + 1] === b) { has = true; break; }
        if (!has) continue;
        addPairs(wi, -1);
        const next = [];
        for (let i = 0; i < w.length; i++) {
          if (i < w.length - 1 && w[i] === a && w[i + 1] === b) { next.push(id); i++; } else next.push(w[i]);
        }
        words[wi] = next;
        addPairs(wi, 1);
      }
      pairCount.delete(bestK);
      if (merges.length % 25 === 0) progress(vocab.length / target);
    }
    return { vocab, merges, dropped };
  }

  // ---------------------------------------------------------------------------
  // The transformer: parameters
  // ---------------------------------------------------------------------------
  const PARAM_NAMES = ["wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
    "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb"];
  const DECAYED = new Set(["wte", "wpe", "qkvw", "attprojw", "fcw", "fcprojw"]);

  function paramSizes(cfg) {
    const { V, T, L, C } = cfg;
    return {
      wte: V * C, wpe: T * C,
      ln1w: L * C, ln1b: L * C, qkvw: L * 3 * C * C, qkvb: L * 3 * C,
      attprojw: L * C * C, attprojb: L * C, ln2w: L * C, ln2b: L * C,
      fcw: L * 4 * C * C, fcb: L * 4 * C, fcprojw: L * C * 4 * C, fcprojb: L * C,
      lnfw: C, lnfb: C,
    };
  }

  function countParams(cfg) {
    const s = paramSizes(cfg);
    return PARAM_NAMES.reduce((a, n) => a + s[n], 0);
  }

  function views(buf, cfg) {
    const s = paramSizes(cfg), v = {};
    let off = 0;
    for (const n of PARAM_NAMES) { v[n] = buf.subarray(off, off + s[n]); off += s[n]; }
    return v;
  }

  // cfg: V vocabulary size, T context length, L layers, NH attention heads, C width.
  function createModel(cfg, params) {
    const total = countParams(cfg);
    const M = {
      cfg, total,
      params: params || new F(total),
      grads: null, m: null, v: null, step: 0,
      acts: new Map(),
    };
    M.P = views(M.params, cfg);
    if (!params) {
      const { L } = cfg;
      const P = M.P;
      const fill = (a, std) => { for (let i = 0; i < a.length; i++) a[i] = randn() * std; };
      fill(P.wte, 0.02); fill(P.wpe, 0.01);
      fill(P.qkvw, 0.02); fill(P.fcw, 0.02);
      // Layers that add into the residual stream start smaller, as in GPT-2.
      fill(P.attprojw, 0.02 / Math.sqrt(2 * L)); fill(P.fcprojw, 0.02 / Math.sqrt(2 * L));
      P.ln1w.fill(1); P.ln2w.fill(1); P.lnfw.fill(1);
    }
    return M;
  }

  function ensureTrainState(M) {
    if (M.grads) return;
    M.grads = new F(M.total);
    M.G = views(M.grads, M.cfg);
    M.m = new Float32Array(M.total);
    M.v = new Float32Array(M.total);
    // 1 where weight decay applies (matrices and embeddings), 0 for biases and LayerNorm.
    M.decay = new Uint8Array(M.total);
    const s = paramSizes(M.cfg);
    let off = 0;
    for (const n of PARAM_NAMES) { if (DECAYED.has(n)) M.decay.fill(1, off, off + s[n]); off += s[n]; }
  }

  // Activations: everything the forward pass computes, kept for the backward pass.
  function getActs(M, key, B, T, withGrads) {
    let A = M.acts.get(key);
    if (A && A.B >= B && A.T >= T && (!withGrads || A.d)) return A;
    const { L, NH, C, V } = M.cfg;
    const BT = B * T;
    const sizes = {
      encoded: BT * C, ln1: L * BT * C, ln1m: L * BT, ln1r: L * BT, qkv: L * BT * 3 * C,
      atty: L * BT * C, att: L * B * NH * T * T, attproj: L * BT * C, res2: L * BT * C,
      ln2: L * BT * C, ln2m: L * BT, ln2r: L * BT, fch: L * BT * 4 * C, fchg: L * BT * 4 * C,
      fcproj: L * BT * C, res3: L * BT * C, lnf: BT * C, lnfm: BT, lnfr: BT,
      logits: BT * V, probs: BT * V,
    };
    A = { B, T };
    for (const k in sizes) A[k] = new F(sizes[k]);
    if (withGrads) {
      A.d = {};
      for (const k of ["encoded", "ln1", "qkv", "atty", "attproj", "res2", "ln2", "fch", "fchg", "fcproj", "res3", "lnf", "logits"]) {
        A.d[k] = new F(sizes[k]);
      }
    }
    A.scratch1 = new F(T); A.scratch2 = new F(T);
    M.acts.set(key, A);
    return A;
  }

  // ---------------------------------------------------------------------------
  // The layers, forward and backward
  // ---------------------------------------------------------------------------
  // Each token becomes (its token vector) + (the vector for its position).
  function encoderForward(out, inp, wte, wpe, B, T, C) {
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
      const o = (b * T + t) * C, w = inp[b * T + t] * C, p = t * C;
      for (let i = 0; i < C; i++) out[o + i] = wte[w + i] + wpe[p + i];
    }
  }
  function encoderBackward(dwte, dwpe, dout, inp, B, T, C) {
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
      const o = (b * T + t) * C, w = inp[b * T + t] * C, p = t * C;
      for (let i = 0; i < C; i++) { const d = dout[o + i]; dwte[w + i] += d; dwpe[p + i] += d; }
    }
  }

  function layernormForward(out, mean, rstd, inp, w, bias, N, C) {
    for (let n = 0; n < N; n++) {
      const x = n * C;
      let m = 0;
      for (let i = 0; i < C; i++) m += inp[x + i];
      m /= C;
      let v = 0;
      for (let i = 0; i < C; i++) { const d = inp[x + i] - m; v += d * d; }
      const s = 1 / Math.sqrt(v / C + 1e-5);
      for (let i = 0; i < C; i++) out[x + i] = (inp[x + i] - m) * s * w[i] + bias[i];
      mean[n] = m; rstd[n] = s;
    }
  }
  function layernormBackward(dinp, dw, db, dout, inp, w, mean, rstd, N, C) {
    for (let n = 0; n < N; n++) {
      const x = n * C, m = mean[n], s = rstd[n];
      let dm = 0, dnm = 0;
      for (let i = 0; i < C; i++) {
        const norm = (inp[x + i] - m) * s, dn = w[i] * dout[x + i];
        dm += dn; dnm += dn * norm;
      }
      dm /= C; dnm /= C;
      for (let i = 0; i < C; i++) {
        const norm = (inp[x + i] - m) * s, dn = w[i] * dout[x + i];
        db[i] += dout[x + i];
        dw[i] += norm * dout[x + i];
        dinp[x + i] += (dn - dm - norm * dnm) * s;
      }
    }
  }

  // out[n, o] = bias[o] + sum_i inp[n, i] * w[o, i]. Four rows at a time, so each
  // weight is read once for four outputs.
  function matmulForward(out, inp, w, bias, N, C, OC) {
    let n = 0;
    for (; n + 4 <= N; n += 4) {
      const i0 = n * C, i1 = i0 + C, i2 = i1 + C, i3 = i2 + C;
      const o0 = n * OC, o1 = o0 + OC, o2 = o1 + OC, o3 = o2 + OC;
      for (let o = 0; o < OC; o++) {
        const wo = o * C;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (let i = 0; i < C; i++) {
          const wv = w[wo + i];
          s0 += inp[i0 + i] * wv; s1 += inp[i1 + i] * wv; s2 += inp[i2 + i] * wv; s3 += inp[i3 + i] * wv;
        }
        const bv = bias ? bias[o] : 0;
        out[o0 + o] = s0 + bv; out[o1 + o] = s1 + bv; out[o2 + o] = s2 + bv; out[o3 + o] = s3 + bv;
      }
    }
    for (; n < N; n++) {
      const x = n * C, y = n * OC;
      for (let o = 0; o < OC; o++) {
        const wo = o * C;
        let s = bias ? bias[o] : 0;
        for (let i = 0; i < C; i++) s += inp[x + i] * w[wo + i];
        out[y + o] = s;
      }
    }
  }
  function matmulBackward(dinp, dw, db, dout, inp, w, N, C, OC) {
    // dinp[n, :] += sum_o dout[n, o] * w[o, :]
    for (let n = 0; n < N; n++) {
      const x = n * C, y = n * OC;
      let o = 0;
      for (; o + 4 <= OC; o += 4) {
        const d0 = dout[y + o], d1 = dout[y + o + 1], d2 = dout[y + o + 2], d3 = dout[y + o + 3];
        const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        for (let i = 0; i < C; i++) dinp[x + i] += d0 * w[w0 + i] + d1 * w[w1 + i] + d2 * w[w2 + i] + d3 * w[w3 + i];
      }
      for (; o < OC; o++) {
        const d = dout[y + o], wo = o * C;
        for (let i = 0; i < C; i++) dinp[x + i] += d * w[wo + i];
      }
    }
    // dw[o, :] += sum_n dout[n, o] * inp[n, :]
    for (let o = 0; o < OC; o++) {
      const wo = o * C;
      let n = 0, sb = 0;
      for (; n + 4 <= N; n += 4) {
        const d0 = dout[n * OC + o], d1 = dout[(n + 1) * OC + o], d2 = dout[(n + 2) * OC + o], d3 = dout[(n + 3) * OC + o];
        sb += d0 + d1 + d2 + d3;
        const x0 = n * C, x1 = x0 + C, x2 = x1 + C, x3 = x2 + C;
        for (let i = 0; i < C; i++) dw[wo + i] += d0 * inp[x0 + i] + d1 * inp[x1 + i] + d2 * inp[x2 + i] + d3 * inp[x3 + i];
      }
      for (; n < N; n++) {
        const d = dout[n * OC + o], x = n * C;
        sb += d;
        for (let i = 0; i < C; i++) dw[wo + i] += d * inp[x + i];
      }
      if (db) db[o] += sb;
    }
  }

  // Causal self-attention. inp holds query, key and value vectors for every
  // token (3C numbers). Each token can only look at itself and earlier tokens.
  function attentionForward(out, att, inp, B, T, C, NH, pre) {
    const C3 = 3 * C, hs = C / NH, scale = 1 / Math.sqrt(hs);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < NH; h++) {
      const q = b * T * C3 + t * C3 + h * hs;
      const a = b * NH * T * T + h * T * T + t * T;
      let max = -Infinity;
      for (let t2 = 0; t2 <= t; t2++) {
        const k = b * T * C3 + t2 * C3 + h * hs + C;
        let s = 0;
        for (let i = 0; i < hs; i++) s += inp[q + i] * inp[k + i];
        s *= scale;
        pre[t2] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let t2 = 0; t2 <= t; t2++) { const e = Math.exp(pre[t2] - max); att[a + t2] = e; sum += e; }
      const inv = 1 / sum;
      for (let t2 = 0; t2 < T; t2++) att[a + t2] = t2 <= t ? att[a + t2] * inv : 0;
      const o = b * T * C + t * C + h * hs;
      for (let i = 0; i < hs; i++) out[o + i] = 0;
      for (let t2 = 0; t2 <= t; t2++) {
        const v = b * T * C3 + t2 * C3 + h * hs + 2 * C, p = att[a + t2];
        for (let i = 0; i < hs; i++) out[o + i] += p * inp[v + i];
      }
    }
  }
  function attentionBackward(dinp, dout, inp, att, B, T, C, NH, datt, dpre) {
    const C3 = 3 * C, hs = C / NH, scale = 1 / Math.sqrt(hs);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) for (let h = 0; h < NH; h++) {
      const a = b * NH * T * T + h * T * T + t * T;
      const o = b * T * C + t * C + h * hs;
      const q = b * T * C3 + t * C3 + h * hs;
      // through "out = sum of att * value"
      for (let t2 = 0; t2 <= t; t2++) {
        const v = b * T * C3 + t2 * C3 + h * hs + 2 * C, p = att[a + t2];
        let s = 0;
        for (let i = 0; i < hs; i++) { s += inp[v + i] * dout[o + i]; dinp[v + i] += p * dout[o + i]; }
        datt[t2] = s;
      }
      // through the softmax
      let dot = 0;
      for (let t2 = 0; t2 <= t; t2++) dot += att[a + t2] * datt[t2];
      for (let t2 = 0; t2 <= t; t2++) dpre[t2] = att[a + t2] * (datt[t2] - dot) * scale;
      // through "score = query . key"
      for (let t2 = 0; t2 <= t; t2++) {
        const k = b * T * C3 + t2 * C3 + h * hs + C, g = dpre[t2];
        for (let i = 0; i < hs; i++) { dinp[q + i] += inp[k + i] * g; dinp[k + i] += inp[q + i] * g; }
      }
    }
  }

  const GELU_S = Math.sqrt(2 / Math.PI);
  function geluForward(out, inp, N) {
    for (let i = 0; i < N; i++) {
      const x = inp[i];
      out[i] = 0.5 * x * (1 + Math.tanh(GELU_S * (x + 0.044715 * x * x * x)));
    }
  }
  function geluBackward(dinp, inp, dout, N) {
    for (let i = 0; i < N; i++) {
      const x = inp[i], u = GELU_S * (x + 0.044715 * x * x * x);
      const th = Math.tanh(u), sech2 = 1 - th * th;
      const g = 0.5 * (1 + th) + x * 0.5 * sech2 * GELU_S * (1 + 3 * 0.044715 * x * x);
      dinp[i] += g * dout[i];
    }
  }

  function residualForward(out, a, b, N) { for (let i = 0; i < N; i++) out[i] = a[i] + b[i]; }
  function residualBackward(da, db, dout, N) { for (let i = 0; i < N; i++) { da[i] += dout[i]; db[i] += dout[i]; } }

  function softmaxRows(out, logits, N, V, temp) {
    const inv = 1 / (temp || 1);
    for (let n = 0; n < N; n++) {
      const x = n * V;
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (logits[x + i] > max) max = logits[x + i];
      let sum = 0;
      for (let i = 0; i < V; i++) { const e = Math.exp((logits[x + i] - max) * inv); out[x + i] = e; sum += e; }
      for (let i = 0; i < V; i++) out[x + i] /= sum;
    }
  }

  function layer(arr, l, size) { return arr.subarray(l * size, (l + 1) * size); }
  function layerParams(P, l, C) {
    return {
      ln1w: layer(P.ln1w, l, C), ln1b: layer(P.ln1b, l, C),
      qkvw: layer(P.qkvw, l, 3 * C * C), qkvb: layer(P.qkvb, l, 3 * C),
      attprojw: layer(P.attprojw, l, C * C), attprojb: layer(P.attprojb, l, C),
      ln2w: layer(P.ln2w, l, C), ln2b: layer(P.ln2b, l, C),
      fcw: layer(P.fcw, l, 4 * C * C), fcb: layer(P.fcb, l, 4 * C),
      fcprojw: layer(P.fcprojw, l, 4 * C * C), fcprojb: layer(P.fcprojb, l, C),
    };
  }
  function layerActs(A, l, B, T, C, NH) {
    const BT = B * T;
    return {
      ln1: layer(A.ln1, l, BT * C), ln1m: layer(A.ln1m, l, BT), ln1r: layer(A.ln1r, l, BT),
      qkv: layer(A.qkv, l, BT * 3 * C), atty: layer(A.atty, l, BT * C), att: layer(A.att, l, B * NH * T * T),
      attproj: layer(A.attproj, l, BT * C), res2: layer(A.res2, l, BT * C),
      ln2: layer(A.ln2, l, BT * C), ln2m: layer(A.ln2m, l, BT), ln2r: layer(A.ln2r, l, BT),
      fch: layer(A.fch, l, BT * 4 * C), fchg: layer(A.fchg, l, BT * 4 * C),
      fcproj: layer(A.fcproj, l, BT * C), res3: layer(A.res3, l, BT * C),
    };
  }

  // Runs the model on B rows of T tokens. Fills A.logits and A.probs: for every
  // position, a probability for every token in the vocabulary. With lastOnly,
  // only the last position of row 0 gets them (enough for generating text).
  function forward(M, A, inp, B, T, lastOnly) {
    const { L, NH, C, V } = M.cfg, P = M.P, BT = B * T;
    encoderForward(A.encoded, inp, P.wte, P.wpe, B, T, C);
    let residual = A.encoded;
    for (let l = 0; l < L; l++) {
      const w = layerParams(P, l, C), a = layerActs(A, l, B, T, C, NH);
      layernormForward(a.ln1, a.ln1m, a.ln1r, residual, w.ln1w, w.ln1b, BT, C);
      matmulForward(a.qkv, a.ln1, w.qkvw, w.qkvb, BT, C, 3 * C);
      attentionForward(a.atty, a.att, a.qkv, B, T, C, NH, A.scratch1);
      matmulForward(a.attproj, a.atty, w.attprojw, w.attprojb, BT, C, C);
      residualForward(a.res2, residual, a.attproj, BT * C);
      layernormForward(a.ln2, a.ln2m, a.ln2r, a.res2, w.ln2w, w.ln2b, BT, C);
      matmulForward(a.fch, a.ln2, w.fcw, w.fcb, BT, C, 4 * C);
      geluForward(a.fchg, a.fch, BT * 4 * C);
      matmulForward(a.fcproj, a.fchg, w.fcprojw, w.fcprojb, BT, 4 * C, C);
      residualForward(a.res3, a.res2, a.fcproj, BT * C);
      residual = a.res3;
    }
    layernormForward(A.lnf, A.lnfm, A.lnfr, residual, P.lnfw, P.lnfb, BT, C);
    if (lastOnly) {
      // The output layer reuses the token vectors (weight tying, as in GPT-2).
      matmulForward(A.logits, A.lnf.subarray((T - 1) * C, T * C), P.wte, null, 1, C, V);
      softmaxRows(A.probs, A.logits, 1, V, 1);
    } else {
      matmulForward(A.logits, A.lnf, P.wte, null, BT, C, V);
      softmaxRows(A.probs, A.logits, BT, V, 1);
    }
  }

  // Backpropagation. A.d.logits must already hold d(loss)/d(logits).
  // Adds d(loss)/d(parameter) into M.grads.
  function backward(M, A, inp, B, T) {
    const { L, NH, C, V } = M.cfg, P = M.P, G = M.G, D = A.d, BT = B * T;
    for (const k in D) if (k !== "logits") D[k].fill(0);
    matmulBackward(D.lnf, G.wte, null, D.logits, A.lnf, P.wte, BT, C, V);
    const lastRes = L > 0 ? layer(A.res3, L - 1, BT * C) : A.encoded;
    const dLastRes = L > 0 ? layer(D.res3, L - 1, BT * C) : D.encoded;
    layernormBackward(dLastRes, G.lnfw, G.lnfb, D.lnf, lastRes, P.lnfw, A.lnfm, A.lnfr, BT, C);
    for (let l = L - 1; l >= 0; l--) {
      const residual = l === 0 ? A.encoded : layer(A.res3, l - 1, BT * C);
      const dres = l === 0 ? D.encoded : layer(D.res3, l - 1, BT * C);
      const w = layerParams(P, l, C), g = layerParams(G, l, C);
      const a = layerActs(A, l, B, T, C, NH);
      const d = {
        ln1: layer(D.ln1, l, BT * C), qkv: layer(D.qkv, l, BT * 3 * C), atty: layer(D.atty, l, BT * C),
        attproj: layer(D.attproj, l, BT * C), res2: layer(D.res2, l, BT * C), ln2: layer(D.ln2, l, BT * C),
        fch: layer(D.fch, l, BT * 4 * C), fchg: layer(D.fchg, l, BT * 4 * C),
        fcproj: layer(D.fcproj, l, BT * C), res3: layer(D.res3, l, BT * C),
      };
      residualBackward(d.res2, d.fcproj, d.res3, BT * C);
      matmulBackward(d.fchg, g.fcprojw, g.fcprojb, d.fcproj, a.fchg, w.fcprojw, BT, 4 * C, C);
      geluBackward(d.fch, a.fch, d.fchg, BT * 4 * C);
      matmulBackward(d.ln2, g.fcw, g.fcb, d.fch, a.ln2, w.fcw, BT, C, 4 * C);
      layernormBackward(d.res2, g.ln2w, g.ln2b, d.ln2, a.res2, w.ln2w, a.ln2m, a.ln2r, BT, C);
      residualBackward(dres, d.attproj, d.res2, BT * C);
      matmulBackward(d.atty, g.attprojw, g.attprojb, d.attproj, a.atty, w.attprojw, BT, C, C);
      attentionBackward(d.qkv, d.atty, a.qkv, a.att, B, T, C, NH, A.scratch1, A.scratch2);
      matmulBackward(d.ln1, g.qkvw, g.qkvb, d.qkv, a.ln1, w.qkvw, BT, C, 3 * C);
      layernormBackward(dres, g.ln1w, g.ln1b, d.ln1, residual, w.ln1w, a.ln1m, a.ln1r, BT, C);
    }
    encoderBackward(G.wte, G.wpe, D.encoded, inp, B, T, C);
  }

  // ---------------------------------------------------------------------------
  // Losses
  // ---------------------------------------------------------------------------
  // Pre-training loss: the average of -ln p(true next token). In nats; divide
  // by ln 2 for bits. Also writes d(loss)/d(logits) = (p - onehot) / N.
  function hardLoss(A, targets, N, V, withGrad) {
    let loss = 0;
    const inv = 1 / N, dl = withGrad ? A.d.logits : null;
    for (let n = 0; n < N; n++) {
      const x = n * V, tgt = targets[n];
      loss -= Math.log(Math.max(A.probs[x + tgt], 1e-30));
      if (dl) {
        for (let i = 0; i < V; i++) dl[x + i] = A.probs[x + i] * inv;
        dl[x + tgt] -= inv;
      }
    }
    return loss / N;
  }

  // Distillation loss. p = teacher's distribution, q = student's, both at
  // temperature tau. Cross-entropy H(p, q) = -sum p log q, entropy H(p),
  // and KL(p || q) = H(p, q) - H(p). H(p) does not depend on the student,
  // so both losses give the same gradient: tau^2 * (q - p) / tau.
  function softLoss(S, Tch, N, V, tau, withGrad) {
    const p = Tch.probs, q = S.probs, dl = withGrad ? S.d.logits : null;
    if (tau !== 1) {
      softmaxRows(Tch.probs, Tch.logits, N, V, tau);
      softmaxRows(S.probs, S.logits, N, V, tau);
    }
    let ce = 0, h = 0;
    const g = tau / N;
    for (let n = 0; n < N; n++) {
      const x = n * V;
      for (let i = 0; i < V; i++) {
        const pi = p[x + i], qi = q[x + i];
        if (pi > 0) { ce -= pi * Math.log(Math.max(qi, 1e-30)); h -= pi * Math.log(pi); }
        if (dl) dl[x + i] = g * (qi - pi);
      }
    }
    return { ce: ce / N, h: h / N, kl: (ce - h) / N };
  }

  // ---------------------------------------------------------------------------
  // Optimiser: AdamW with gradient clipping
  // ---------------------------------------------------------------------------
  function adamw(M, lr, wd, clip) {
    const g = M.grads, p = M.params, m = M.m, v = M.v, dec = M.decay;
    let norm = 0;
    for (let i = 0; i < g.length; i++) norm += g[i] * g[i];
    norm = Math.sqrt(norm);
    const scale = clip > 0 && norm > clip ? clip / norm : 1;
    const b1 = 0.9, b2 = 0.95, eps = 1e-8;
    M.step++;
    const c1 = 1 - Math.pow(b1, M.step), c2 = 1 - Math.pow(b2, M.step);
    for (let i = 0; i < p.length; i++) {
      const gi = g[i] * scale;
      const mi = m[i] = b1 * m[i] + (1 - b1) * gi;
      const vi = v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      p[i] -= lr * ((mi / c1) / (Math.sqrt(vi / c2) + eps) + (dec[i] ? wd * p[i] : 0));
    }
    return norm;
  }

  // Warm up, then cosine decay down to 10% of the peak learning rate.
  function lrAt(step, o) {
    if (step < o.warmup) return o.lr * (step + 1) / o.warmup;
    const frac = Math.min(1, (step - o.warmup) / Math.max(1, o.steps - o.warmup));
    return o.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * frac)));
  }

  // ---------------------------------------------------------------------------
  // State and data
  // ---------------------------------------------------------------------------
  const S = {
    tok: null,          // tokenizer
    data: null,         // Uint16Array of every token
    split: 0,           // tokens before this index are training data, after it are validation
    stats: null,
    teacher: null,      // the pre-trained model
    student: null,      // a distilled model
    running: false,
    genId: 0,
  };

  function sampleBatch(B, T, inp, tgt) {
    const max = S.split - T - 1;
    for (let b = 0; b < B; b++) {
      const s = Math.floor(rand() * (max + 1));
      for (let t = 0; t < T; t++) { inp[b * T + t] = S.data[s + t]; tgt[b * T + t] = S.data[s + t + 1]; }
    }
  }

  // Up to `maxRows` windows of validation text, laid end to end.
  function valWindows(T, maxRows) {
    const len = S.data.length - S.split;
    const n = Math.max(1, Math.min(maxRows, Math.floor((len - 1) / T)));
    const step = Math.max(T, Math.floor((len - 1 - T) / Math.max(1, n - 1)));
    const starts = [];
    for (let i = 0; i < n; i++) starts.push(S.split + Math.min(i * step, len - 1 - T));
    return starts;
  }

  function evalVal(M, B, maxRows) {
    const T = M.cfg.T, V = M.cfg.V;
    const A = getActs(M, "eval", B, T, false);
    const starts = valWindows(T, maxRows);
    const inp = new Int32Array(B * T), tgt = new Int32Array(B * T);
    let total = 0, count = 0;
    for (let i = 0; i < starts.length; i += B) {
      const rows = Math.min(B, starts.length - i);
      for (let r = 0; r < rows; r++) for (let t = 0; t < T; t++) {
        inp[r * T + t] = S.data[starts[i + r] + t]; tgt[r * T + t] = S.data[starts[i + r] + t + 1];
      }
      forward(M, A, inp, rows, T);
      total += hardLoss(A, tgt, rows * T, V, false) * rows * T;
      count += rows * T;
    }
    return total / count;
  }

  async function gzipBytes(str) {
    if (typeof CompressionStream === "undefined") return null;
    const stream = new Blob([str]).stream().pipeThrough(new CompressionStream("gzip"));
    return (await new Response(stream).arrayBuffer()).byteLength;
  }

  function showPiece(s) { return s.replace(/\n/g, "↵").replace(/\t/g, "⇥"); }

  // ---------------------------------------------------------------------------
  // Message handlers
  // ---------------------------------------------------------------------------
  async function tokenize(m) {
    const t0 = performance.now();
    S.running = false;
    const text = m.text.replace(/\r\n?/g, "\n");
    let tok, dropped = 0;
    const reuse = m.reuse && S.tok && S.teacher;
    if (reuse) {
      tok = S.tok;            // keep training the current model on new text
    } else if (m.kind === "bpe") {
      post({ type: "progress", pct: 0, phase: "Learning subword merges (BPE)…" });
      const r = trainBPE(text, m.vocabSize, f => post({ type: "progress", pct: f * 60, phase: `Learning subword merges (BPE)… ${Math.round(f * 100)}%` }));
      dropped = r.dropped;
      tok = makeTokenizer("bpe", r.vocab, r.merges);
    } else {
      const r = baseVocab(text);
      dropped = r.dropped;
      tok = makeTokenizer("char", r.vocab, []);
    }
    post({ type: "progress", pct: 60, phase: "Turning the text into token ids…" });
    const data = tok.encode(text, f => post({ type: "progress", pct: 60 + f * 30, phase: `Turning the text into token ids… ${Math.round(f * 100)}%` }));
    if (data.length < 200) throw new Error("The text is too short. Give it at least a few hundred words.");
    if (!reuse) S.teacher = S.student = null;
    S.tok = tok;
    S.data = data;
    S.split = Math.floor(data.length * 0.9);

    // Baselines, in bits per token, measured on the validation text.
    post({ type: "progress", pct: 92, phase: "Measuring baselines (gzip, word counts)…" });
    const V = tok.size;
    const counts = new Float64Array(V);
    for (let i = 0; i < S.split; i++) counts[data[i]]++;
    let unigramVal = 0;
    for (let i = S.split; i < data.length; i++) unigramVal -= Math.log2((counts[data[i]] + 1) / (S.split + V));
    const valTokens = data.length - S.split;
    unigramVal /= valTokens;
    let unigramH = 0;
    for (let i = 0; i < V; i++) if (counts[i]) { const p = counts[i] / S.split; unigramH -= p * Math.log2(p); }

    const valText = tok.decode(data.subarray(S.split));
    const trainTail = tok.decode(data.subarray(Math.max(0, S.split - 40000), S.split)).slice(-32000);
    const gzVal = await gzipBytes(valText);
    let gzCond = null;
    if (gzVal != null) {
      // The "zipping" trick: how many extra bytes the validation text costs when
      // gzip has just seen the end of the training text.
      gzCond = (await gzipBytes(trainTail + valText)) - (await gzipBytes(trainTail));
    }
    const counted = [];
    for (let i = 0; i < V; i++) counted.push([i, counts[i]]);
    counted.sort((a, b) => b[1] - a[1]);

    S.stats = {
      kind: tok.kind, reused: !!reuse, chars: text.length, tokens: data.length, vocab: V, merges: tok.merges.length, dropped,
      trainTokens: S.split, valTokens, valChars: valText.length,
      uniformBits: Math.log2(V), unigramBits: unigramVal, unigramEntropy: unigramH,
      gzipBytes: gzVal, gzipCondBytes: gzCond,
      preview: Array.from(data.subarray(0, 80), id => tok.vocab[id]),
      longest: tok.vocab.slice(tok.vocab.length - tok.merges.length).sort((a, b) => b.length - a.length).slice(0, 12),
      common: counted.slice(0, 16).filter(x => x[1] > 0).map(([id, c]) => [tok.vocab[id], c]),
      ms: performance.now() - t0,
    };
    post({ type: "tokenized", stats: S.stats, id: m.id });
  }

  function modelInfo(M) {
    return { cfg: M.cfg, params: M.total, step: M.step };
  }

  function create(m) {
    if (!S.tok) throw new Error("Tokenize some text first.");
    if (m.cfg.C % m.cfg.NH !== 0) throw new Error("Width must be divisible by the number of heads.");
    seed = m.seed || 1337;
    const cfg = { V: S.tok.size, T: m.cfg.T, L: m.cfg.L, NH: m.cfg.NH, C: m.cfg.C };
    if (S.data.length - S.split < cfg.T + 2) throw new Error("The validation text is shorter than the context length. Use a shorter context or more text.");
    S.teacher = createModel(cfg);
    S.student = null;
    const init = evalVal(S.teacher, 8, 16);
    S.teacher.lastVal = init / LN2;
    post({ type: "created", info: modelInfo(S.teacher), initLoss: init / LN2, id: m.id });
  }

  // Pre-training, or distillation (m.distill) of a smaller student model.
  // o.mode for distillation: "ce" (cross-entropy with the teacher's
  // distribution), "kl" (KL divergence from it), or "hard" (the true next
  // tokens only, so no teacher: a baseline). Runs o.steps steps, or until a
  // "stop" message. With m.resume it carries on with the same schedule.
  async function train(m) {
    const o = m.opts;
    const distill = !!m.distill;
    if (!S.data) throw new Error("Tokenize some text first.");
    let M;
    if (distill) {
      if (!S.teacher) throw new Error("Pre-train a teacher model first.");
      if (!m.resume || !S.student) {
        seed = (m.seed || 1337) + 1;
        const t = S.teacher.cfg;
        S.student = createModel({ V: t.V, T: t.T, L: m.student.L, NH: m.student.NH, C: m.student.C });
      }
      M = S.student;
    } else {
      M = S.teacher;
      if (!M) throw new Error("Create a model first.");
    }
    ensureTrainState(M);
    const { T, V } = M.cfg, B = o.B;
    const useTeacher = distill && o.mode !== "hard";
    const A = getActs(M, "train", B, T, true);
    const TA = useTeacher ? getActs(S.teacher, "teach", B, T, false) : null;
    const inp = new Int32Array(B * T), tgt = new Int32Array(B * T);
    const kind = distill ? "d" : "t";
    const fresh = !m.resume || M.schedEnd == null || M.step >= M.schedEnd;
    if (fresh) { M.schedStart = M.step; M.schedEnd = M.step + o.steps; }
    const sched = { lr: o.lr, warmup: o.warmup, steps: M.schedEnd - M.schedStart };
    S.running = true;
    post({ type: kind + "start", info: modelInfo(M), id: m.id });
    if (distill && S.teacher.lastVal == null) S.teacher.lastVal = evalVal(S.teacher, B, o.evalRows) / LN2;
    if (fresh) {
      M.lastVal = evalVal(M, B, o.evalRows) / LN2;
      post({ type: kind + "eval", step: M.step, val: M.lastVal, teacherVal: distill ? S.teacher.lastVal : undefined });
    }
    let lastPost = 0, tokens = 0;
    const tStart = performance.now();
    while (S.running && M.step < M.schedEnd) {
      const t0 = performance.now();
      sampleBatch(B, T, inp, tgt);
      forward(M, A, inp, B, T);
      const hard = hardLoss(A, tgt, B * T, V, !useTeacher);
      let soft = null;
      if (useTeacher) {
        forward(S.teacher, TA, inp, B, T);
        soft = softLoss(A, TA, B * T, V, o.tau, true);
      }
      M.grads.fill(0);
      backward(M, A, inp, B, T);
      const lr = lrAt(M.step - M.schedStart, sched);
      const gnorm = adamw(M, lr, o.wd, 1.0);
      tokens += B * T;
      const now = performance.now();
      const done = M.step - M.schedStart, last = M.step >= M.schedEnd;
      if (now - lastPost > 60 || last) {
        const msg = {
          type: kind + "step", step: M.step, done, total: sched.steps, loss: hard / LN2, lr, gnorm,
          ms: now - t0, tokPerSec: tokens / ((now - tStart) / 1000),
        };
        if (soft) { msg.ce = soft.ce / LN2; msg.kl = soft.kl / LN2; msg.h = soft.h / LN2; }
        post(msg);
        lastPost = now;
      }
      if (done % o.evalEvery === 0 || last) {
        M.lastVal = evalVal(M, B, o.evalRows) / LN2;
        post({ type: kind + "eval", step: M.step, val: M.lastVal, teacherVal: distill ? S.teacher.lastVal : undefined });
      }
      if (!last) await yieldNow();
    }
    S.running = false;
    post({ type: kind + "done", info: modelInfo(M), finished: M.step >= M.schedEnd, id: m.id });
  }

  function pickModel(which) {
    const M = which === "student" ? S.student : S.teacher;
    if (!M) throw new Error(which === "student" ? "Distill a student model first." : "Create a model first.");
    return M;
  }

  function topList(probs, V, k) {
    const idx = [];
    for (let i = 0; i < V; i++) idx.push(i);
    idx.sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k).map(i => ({ s: S.tok.vocab[i], p: probs[i] }));
  }

  function entropyBits(probs, V) {
    let h = 0;
    for (let i = 0; i < V; i++) if (probs[i] > 0) h -= probs[i] * Math.log2(probs[i]);
    return h;
  }

  async function generate(m) {
    const M = pickModel(m.which);
    const { T, V } = M.cfg;
    const A = getActs(M, "gen", 1, T, false);
    const id = ++S.genId;
    let ids = Array.from(S.tok.encode(m.prompt || ""));
    if (!ids.length) {
      const nl = S.tok.vocab.indexOf("\n");
      ids = [nl >= 0 ? nl : S.data[Math.floor(rand() * S.split)]];
    }
    const promptLen = ids.length;
    const inp = new Int32Array(T);
    const sampled = new Float64Array(V);
    for (let n = 0; n < m.n && S.genId === id; n++) {
      const ctx = ids.slice(-T);
      inp.set(ctx);
      forward(M, A, inp, 1, ctx.length, true);
      const probs = A.probs.subarray(0, V);
      // Temperature and top-k change how we pick, not what the model believes.
      let next;
      if (m.temp <= 0.001) {
        next = 0;
        for (let i = 1; i < V; i++) if (probs[i] > probs[next]) next = i;
      } else {
        let cut = 0;
        if (m.topK > 0 && m.topK < V) cut = Array.from(probs).sort((a, b) => b - a)[m.topK - 1];
        let sum = 0;
        for (let i = 0; i < V; i++) {
          sampled[i] = probs[i] >= cut ? Math.pow(probs[i], 1 / m.temp) : 0;
          sum += sampled[i];
        }
        let r = rand() * sum;
        next = V - 1;
        for (let i = 0; i < V; i++) { r -= sampled[i]; if (r <= 0) { next = i; break; } }
      }
      post({
        type: "gen", id: m.id, s: S.tok.vocab[next], p: probs[next], bits: -Math.log2(Math.max(probs[next], 1e-30)),
        entropy: entropyBits(probs, V), top: topList(probs, V, 8), ctx: ctx.length, cut: ids.length > T,
      });
      ids.push(next);
      await yieldNow();
    }
    post({ type: "gendone", id: m.id, promptLen });
  }

  // For each token of a text: the chance the model gave it, and its information
  // content -log2 p. The average of those is the loss.
  function analyze(m) {
    const M = pickModel(m.which);
    const { T, V } = M.cfg;
    let ids;
    if (m.text && m.text.trim()) ids = Array.from(S.tok.encode(m.text.replace(/\r\n?/g, "\n")));
    else if (!S.data) throw new Error("Type some text to score.");
    else ids = Array.from(S.data.subarray(S.split, Math.min(S.data.length, S.split + 300)));
    ids = ids.slice(0, 600);
    if (ids.length < 2) throw new Error("Type a longer text to score.");
    const A = getActs(M, "ana", 1, T, false);
    const out = [{ s: S.tok.vocab[ids[0]], first: true }];
    let bits = 0, chars = 0;
    // Slide a window so every prediction sees up to T tokens before it.
    for (let i = 1; i < ids.length; i++) {
      const ctx = ids.slice(Math.max(0, i - T), i);
      forward(M, A, Int32Array.from(ctx), 1, ctx.length, true);
      const probs = A.probs.subarray(0, V);
      const p = probs[ids[i]], b = -Math.log2(Math.max(p, 1e-30));
      bits += b; chars += S.tok.vocab[ids[i]].length;
      out.push({ s: S.tok.vocab[ids[i]], p, bits: b, top: topList(probs, V, 5) });
    }
    const n = Math.max(1, ids.length - 1);
    post({ type: "analysis", id: m.id, tokens: out, avgBits: bits / n, bitsPerChar: bits / Math.max(1, chars), which: m.which });
  }

  // A tiny teaching example: the model's beliefs about what comes next.
  function predict(m) {
    const M = pickModel(m.which);
    const { T, V } = M.cfg;
    const A = getActs(M, "gen", 1, T, false);
    let ids = Array.from(S.tok.encode(m.text || ""));
    if (!ids.length) ids = [S.tok.vocab.indexOf("\n") >= 0 ? S.tok.vocab.indexOf("\n") : 0];
    const ctx = ids.slice(-T);
    forward(M, A, Int32Array.from(ctx), 1, ctx.length, true);
    const probs = A.probs.subarray(0, V);
    post({ type: "prediction", id: m.id, top: topList(probs, V, 12), entropy: entropyBits(probs, V) });
  }

  function exportModel(m) {
    const M = pickModel(m.which);
    const params = new Float32Array(M.params);
    post({
      type: "exported", id: m.id,
      model: { format: "anything-to-llm/gpt/1", cfg: M.cfg, step: M.step, tokenizer: { kind: S.tok.kind, vocab: S.tok.vocab, merges: S.tok.merges } },
      params: params.buffer,
    }, [params.buffer]);
  }

  function importModel(m) {
    const meta = m.model;
    if (!meta || meta.format !== "anything-to-llm/gpt/1") throw new Error("This is not a model file from this page.");
    const tok = makeTokenizer(meta.tokenizer.kind, meta.tokenizer.vocab, meta.tokenizer.merges);
    const params = new F(new Float32Array(m.params));
    if (params.length !== countParams(meta.cfg)) throw new Error("The model file is damaged (wrong number of parameters).");
    S.running = false;
    S.tok = tok;
    S.data = null;
    S.teacher = createModel(meta.cfg, params);
    S.teacher.step = meta.step || 0;
    S.student = null;
    post({ type: "imported", id: m.id, info: modelInfo(S.teacher), tokenizer: { kind: tok.kind, vocab: tok.size, merges: tok.merges.length } });
  }

  const handlers = { tokenize, create, train, generate, analyze, predict, export: exportModel, import: importModel };

  self.onmessage = async (e) => {
    const m = e.data;
    if (m.type === "stop") { S.running = false; return; }
    if (m.type === "stopgen") { S.genId++; return; }
    const h = handlers[m.type];
    if (!h) return;
    try {
      await h(m);
    } catch (err) {
      S.running = false;
      post({ type: "error", id: m.id, op: m.type, message: err && err.message ? err.message : String(err) });
    }
  };

  // For tests in Node.js.
  self.__nn = { createModel, forward, backward, hardLoss, softLoss, getActs, ensureTrainState, trainBPE, makeTokenizer, countParams, S, adamw };
}
