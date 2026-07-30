import assert from "node:assert/strict";
import test from "node:test";

import { PROTOCOL_VERSION, WorkerClient } from "../lib/worker-protocol.js";

class FakeWorker {
  static instances = [];

  constructor() {
    this.listeners = new Map();
    this.messages = [];
    this.terminated = false;
    FakeWorker.instances.push(this);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  postMessage(message) {
    this.messages.push(message);
  }

  terminate() {
    this.terminated = true;
  }

  emit(type, event) {
    for (const listener of this.listeners.get(type) || []) listener(event);
  }
}

test("fatal worker errors reject current and future work; retry needs a fresh worker", async () => {
  const previousWorker = globalThis.Worker;
  globalThis.Worker = FakeWorker;
  try {
    const client = new WorkerClient({
      url: "/fake-worker.js",
      name: "fatal-test",
      maxInFlight: 1,
      maxQueue: 0,
      disposeGraceMs: 0,
    });
    const firstWorker = FakeWorker.instances.at(-1);
    firstWorker.emit("message", { data: { p: PROTOCOL_VERSION, kind: "ready" } });
    await client.ready;

    const current = client.request("classify", { audio: new ArrayBuffer(4) });
    firstWorker.emit("error", { message: "forced fatal error" });
    await assert.rejects(current, /forced fatal error/);
    assert.equal(client.pending, 0);
    await assert.rejects(
      client.request("classify", {}),
      /is (error|terminated); create a new worker before retrying/,
    );
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(firstWorker.terminated, true);

    const retry = new WorkerClient({
      url: "/fake-worker.js",
      name: "retry-test",
      maxInFlight: 1,
      maxQueue: 0,
      disposeGraceMs: 0,
    });
    const secondWorker = FakeWorker.instances.at(-1);
    assert.notEqual(secondWorker, firstWorker);
    secondWorker.emit("message", { data: { p: PROTOCOL_VERSION, kind: "ready" } });
    await retry.ready;

    const retried = retry.request("classify", {});
    const request = secondWorker.messages.find((message) => message.kind === "request");
    secondWorker.emit("message", {
      data: { p: PROTOCOL_VERSION, kind: "response", id: request.id, result: { ok: true } },
    });
    assert.deepEqual(await retried, { result: { ok: true } });
    await retry.terminate();
  } finally {
    globalThis.Worker = previousWorker;
    FakeWorker.instances.length = 0;
  }
});
