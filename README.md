# Anything → LLM

Turn any text into a tiny language model you can chat with. It is a fun way to learn how large language models work.

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

## Files

- `index.html`: the page
- `style.css`: the styles (light and dark mode)
- `app.js`: the UI and the model (the model runs in a Web Worker made from a Blob)
- `samples.js`: the built-in sample texts

There is no build step and there are no dependencies.

## Turn on GitHub Pages

1. Go to the repository's **Settings → Pages**.
2. Under **Source**, pick **Deploy from a branch**.
3. Pick the branch (for example `main`) and the `/ (root)` folder, then save.

You can also open `index.html` directly from your computer. It works offline.
