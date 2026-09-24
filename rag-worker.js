// Runs the language model in a Web Worker, so downloading the model and
// generating text never freeze the page.
import { CreateMLCEngine } from "https://esm.run/@mlc-ai/web-llm@0.2.85";

let engine = null;
let loadedModel = null;

function post(msg) { self.postMessage(msg); }

async function load(model) {
  if (engine && loadedModel === model) { post({ type: "ready", model }); return; }
  const initProgressCallback = (r) => post({ type: "progress", progress: r.progress, text: r.text });
  if (engine) {
    await engine.reload(model);
  } else {
    engine = await CreateMLCEngine(model, { initProgressCallback });
  }
  loadedModel = model;
  post({ type: "ready", model });
}

async function chat(id, messages, options) {
  if (!engine) throw new Error("The AI model is not loaded yet.");
  const stream = await engine.chat.completions.create({
    messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: options.temperature ?? 0.3,
    max_tokens: options.max_tokens ?? 512,
  });
  let usage = null;
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content;
    if (delta) post({ type: "delta", id, text: delta });
    if (chunk.usage) usage = chunk.usage;
  }
  post({ type: "done", id, usage });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") await load(m.model);
    else if (m.type === "chat") await chat(m.id, m.messages, m.options || {});
    else if (m.type === "stop") { if (engine) await engine.interruptGenerate(); }
  } catch (err) {
    post({ type: "error", id: m.id, during: m.type, message: String(err?.message || err) });
  }
};
