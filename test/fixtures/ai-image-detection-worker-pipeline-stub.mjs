// Stands in for /web-ai-showcase/lib/webai.js — and for the transformers.js module that specifier
// loads — when test/ai-image-detection-worker.test.mjs imports the REAL worker module unmodified.
//
// Everything this stub does is recorded on / driven by globalThis.__aiDetectStub, which the test
// installs before importing a fresh copy of the worker. No network, no model weights, no browser,
// no Chrome: the pipeline is made of plain promises, so a test can control forward latency, force
// failures, and observe how many forwards are alive at once.
export const TRANSFORMERS_URL = import.meta.url; // importing this module also yields RawImage below

export class RawImage {
  static async read(url) {
    return globalThis.__aiDetectStub.decode(url);
  }
}

export async function loadPipeline({ onProgress }) {
  return globalThis.__aiDetectStub.load(onProgress);
}
