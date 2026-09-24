// Chunking and search for "Chat with a book".
// Plain ES module, no dependencies. It works in the browser and in Node (for tests).

const CHUNK_MAX_WORDS = 200;   // split paragraphs longer than this
const CHUNK_MIN_WORDS = 60;    // join short paragraphs until at least this long

// Common words that say little about what a passage is about.
const STOPWORDS = new Set((
  "a about above after again against all am an and any are as at be because been before being below between both but by " +
  "can could did do does doing down during each few for from further had has have having he her here hers herself him himself " +
  "his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our ours " +
  "ourselves out over own same she should so some such than that the their theirs them themselves then there these they this " +
  "those through to too under until up very was we were what when where which while who whom why will with would you your " +
  "yours yourself yourselves tell know say said says does did also into upon shall may might must us one ever"
).split(" "));

// Cut off the Project Gutenberg licence text, if there is one.
export function stripGutenberg(text) {
  const start = text.search(/\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n/i);
  if (start >= 0) text = text.slice(text.indexOf("\n", start) + 1);
  const end = text.search(/\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG/i);
  if (end >= 0) text = text.slice(0, end);
  return text;
}

function wordCount(s) {
  const m = s.match(/\S+/g);
  return m ? m.length : 0;
}

// Split one long paragraph into pieces of at most CHUNK_MAX_WORDS, cutting
// between sentences where possible.
function splitLong(par) {
  const sentences = par.match(/[^.!?]+(?:[.!?]+["'’”)\]]*|$)\s*/g) || [par];
  const out = [];
  let cur = [], n = 0;
  for (const s of sentences) {
    const w = s.split(/\s+/).filter(Boolean);
    if (n + w.length > CHUNK_MAX_WORDS && n > 0) { out.push(cur.join(" ")); cur = []; n = 0; }
    // a single sentence that is too long: cut it by word count
    for (let i = 0; i < w.length; i += CHUNK_MAX_WORDS) {
      const part = w.slice(i, i + CHUNK_MAX_WORDS);
      if (n + part.length > CHUNK_MAX_WORDS && n > 0) { out.push(cur.join(" ")); cur = []; n = 0; }
      cur.push(...part); n += part.length;
    }
  }
  if (n) out.push(cur.join(" "));
  return out;
}

// Split text into chunks: whole paragraphs, long ones cut down, short ones joined.
export function chunkText(text) {
  text = text.replace(/\r\n?/g, "\n");
  // Blank lines separate paragraphs. Single line breaks are just line wrapping.
  let paragraphs = text.split(/\n[ \t]*\n+/).map(p => p.replace(/\s+/g, " ").trim()).filter(Boolean);
  // A text with no blank lines at all: fall back to one paragraph per line.
  if (paragraphs.length === 1 && wordCount(paragraphs[0]) > CHUNK_MAX_WORDS && text.includes("\n")) {
    paragraphs = text.split("\n").map(p => p.replace(/\s+/g, " ").trim()).filter(Boolean);
  }

  const chunks = [];
  let buf = "", bufWords = 0;
  const flush = () => { if (bufWords) chunks.push(buf); buf = ""; bufWords = 0; };
  for (const p of paragraphs) {
    const n = wordCount(p);
    if (n > CHUNK_MAX_WORDS) {
      flush();
      for (const piece of splitLong(p)) chunks.push(piece);
      continue;
    }
    if (bufWords && bufWords + n > CHUNK_MAX_WORDS) flush();
    buf = bufWords ? buf + "\n\n" + p : p;
    bufWords += n;
    if (bufWords >= CHUNK_MIN_WORDS) flush();
  }
  flush();
  return chunks;
}

// Very small stemmer so "whales", "whaling" and "whale" match each other.
function stem(w) {
  if (w.length > 5 && w.endsWith("ing")) return w.slice(0, -3);
  if (w.length > 4 && w.endsWith("ed")) return w.slice(0, -2);
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 4 && w.endsWith("es") && /(sh|ch|x|ss)es$/.test(w)) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

export function terms(text) {
  const out = [];
  const re = /[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu;
  let m;
  while ((m = re.exec(text.toLowerCase()))) {
    let w = m[0].replace(/['’]s$/, "");
    if (w.length < 2 || STOPWORDS.has(w)) continue;
    out.push(stem(w));
  }
  return out;
}

// BM25 index: a well-known, improved version of TF-IDF scoring.
// Build it with buildIndex(); it yields to the browser now and then so the page
// keeps responding, and reports progress through onProgress(0..1).
export async function buildIndex(chunks, onProgress = () => {}) {
  const postings = new Map();      // term -> {ids: number[], tfs: number[]}
  const lengths = new Uint32Array(chunks.length);
  let totalLen = 0;
  let lastYield = Date.now();

  for (let i = 0; i < chunks.length; i++) {
    const ts = terms(chunks[i]);
    lengths[i] = ts.length;
    totalLen += ts.length;
    const tf = new Map();
    for (const t of ts) tf.set(t, (tf.get(t) || 0) + 1);
    for (const [t, c] of tf) {
      let p = postings.get(t);
      if (!p) { p = { ids: [], tfs: [] }; postings.set(t, p); }
      p.ids.push(i);
      p.tfs.push(c);
    }
    if (Date.now() - lastYield > 30) {
      onProgress(i / chunks.length);
      await new Promise(r => setTimeout(r, 0));
      lastYield = Date.now();
    }
  }
  onProgress(1);
  return { chunks, postings, lengths, avgLen: totalLen / Math.max(1, chunks.length) };
}

// Find the k chunks that best match the query.
// Returns [{id, score, text, matched: [terms]}], best first.
export function search(index, query, k = 3) {
  const { chunks, postings, lengths, avgLen } = index;
  const N = chunks.length;
  const K1 = 1.2, B = 0.75;
  const qTerms = [...new Set(terms(query))];
  const scores = new Float64Array(N);
  const hits = new Uint16Array(N);   // how many different query words each chunk has

  for (const t of qTerms) {
    const p = postings.get(t);
    if (!p) continue;
    const idf = Math.log(1 + (N - p.ids.length + 0.5) / (p.ids.length + 0.5));
    for (let j = 0; j < p.ids.length; j++) {
      const id = p.ids[j], tf = p.tfs[j];
      scores[id] += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * lengths[id] / avgLen));
      hits[id]++;
    }
  }

  // Keyword overlap bonus: prefer chunks that contain more of the query words.
  const top = [];
  for (let i = 0; i < N; i++) {
    if (scores[i] <= 0) continue;
    const s = scores[i] * (1 + 0.5 * (hits[i] - 1) / Math.max(1, qTerms.length));
    if (top.length < k || s > top[top.length - 1].score) {
      top.push({ id: i, score: s });
      top.sort((a, b) => b.score - a.score);
      if (top.length > k) top.pop();
    }
  }
  return top.map(r => ({
    ...r,
    text: chunks[r.id],
    matched: qTerms.filter(t => postings.get(t)?.ids.includes(r.id)),
  }));
}

export const queryTerms = terms;
