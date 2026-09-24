# Anything → LLM

The site has two pages:

- **Build a tiny LLM** (`index.html`): turn any text into a tiny language model you can chat with. It is a fun way to learn how large language models work.
- **Chat with a book** (`rag.html`): upload a book and ask a real AI about it. A small AI model runs in your browser with [WebLLM](https://github.com/mlc-ai/web-llm), and the page uses retrieval-augmented generation (RAG).

**Live site:** https://hardy02.github.io/Anything-to-llm/ (after GitHub Pages is turned on, see below)

## What it does

1. You type or paste text, upload one or more `.txt` files, or pick a built-in sample. You can use up to **50,000,000 words**. Uploaded files are read piece by piece, so files of hundreds of MB work.
2. The site splits the text into tokens and builds an index that can find every run of words in it, of any length.
3. You chat with it. It picks each next word at random, using the probabilities from those counts, just like an LLM predicts the next token.

To learn more, you can:

- **Click any word in a reply** to see the words the model looked at, the options it had, and the chance of each one.
- **Colour by probability**: see which words were near-certain and which were a gamble.
- **Be the model**: type a few words and see the next-word probabilities live. Click a word to add it.
- Change **Memory** (from 1 word up to 1,024 words, or **Unlimited**), **Creativity** (temperature) and **Avoid copying** to see how the output changes.

Everything runs in your browser, in a Web Worker. Your text is never uploaded.

## How it works (technical)

- Word-level n-gram model with backoff. With **Unlimited** memory it is an "∞-gram" model: it always uses the longest recent run of words that appears somewhere in the text.
- The model builds a suffix array over the token sequence, using prefix doubling with radix sort, sorted to a depth of 1,024 tokens. It stops early when every position is unique. A binary search then finds any context. It also uses binary search to find the longest matching context, because a match for k words means there is also a match for k−1.
- After building, it keeps about 8 bytes per token. The peak during building is about 24 bytes per token.
- Input is streamed: files are decoded in chunks (`File.stream()` + `TextDecoderStream`) inside the worker, and tokens go straight into a growing `Int32Array`.
- Build times in headless Chromium, on a worst-case text with long repeated passages:

  | Words | Build time |
  |------:|-----------:|
  | 5M    | about 7 s  |
  | 25M   | about 46 s |
  | 50M   | about 106 s |

  Normal prose usually builds faster, because most positions become unique within a few words. Phones have less memory, so they may not manage the largest texts.
- It applies temperature to the counts: `p ∝ count^(1/T)`.
- In chat mode, it continues from the end of your message if it has seen those words. If it hasn't, it starts from the rarest word in your message that it knows.

## Chat with a book (RAG)

1. **Load the AI.** Pick a model (Llama 3.2 1B by default, or Qwen 2.5 0.5B, or Llama 3.2 3B). `rag-worker.js` is a module Web Worker. It calls `CreateMLCEngine` from `@mlc-ai/web-llm`, so downloading the model and generating text never freeze the page. The model is cached in the browser after the first download. If the GPU has no `shader-f16` support, the page uses the `q4f32` version of the model.
2. **Upload a book.** The page reads `.txt` files with `FileReader`, removes Project Gutenberg licence text, and splits the book into chunks of 60–200 words. It splits by paragraph, cuts long paragraphs between sentences, and joins short ones.
3. **Search.** `search.js` builds a BM25 index, a stronger version of TF-IDF. It uses stopwords and a small stemmer, and gives a bonus to chunks that contain more of the question's words. It finds the top 3 chunks for each question. A 1-million-word book indexes in about a second.
4. **RAG prompt.** The page builds a hidden system prompt: *"You are a helpful assistant. Answer the user's question ONLY using the following text snippets. If the answer is not in the text, say you do not know."*, followed by the 3 chunks. It also sends the last 2 question/answer pairs, so follow-up questions work.
5. **Streaming.** The answer appears token by token, and a **Stop** button can interrupt it. Under each answer, **Sources** shows the passages the AI was given, with matching words highlighted. It also shows the hidden prompt.

The AI needs a browser with **WebGPU**, such as a recent Chrome or Edge on a computer. Without WebGPU, the page still works as a search engine: questions show the best matching passages.

`rag.html` uses `<script type="module">` and a module worker, so it must be served over HTTP (GitHub Pages is fine). It does not work from `file://`. To test it on your computer, run `python3 -m http.server` and open `http://localhost:8000/rag.html`.

## Files

- `index.html`: the page
- `style.css`: the styles (light and dark mode)
- `app.js`: the UI and the model (the model runs in a Web Worker made from a Blob)
- `samples.js`: the built-in sample texts
- `rag.html`, `rag.css`, `rag.js`: the "Chat with a book" page
- `search.js`: chunking and BM25 search (an ES module with no dependencies)
- `rag-worker.js`: the Web Worker that runs WebLLM (`@mlc-ai/web-llm@0.2.85`, loaded from the jsDelivr CDN)

Script and style links end in `?v=N`. When you change these files, increase `N`, so that browsers don't keep old cached copies next to a new `index.html`.

There is no build step, no Node.js, and no bundler.

## Turn on GitHub Pages

1. Go to the repository's **Settings → Pages**.
2. Under **Source**, pick **Deploy from a branch**.
3. Pick the branch (for example `main`) and the `/ (root)` folder, then save.

You can also open `index.html` directly from your computer. It works offline.
