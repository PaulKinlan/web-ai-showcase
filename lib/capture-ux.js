// capture-ux.js — a reusable, accessible media-capture component for the web-ai-showcase demos.
//
// One component covers the four ways a demo takes a media input from the visitor:
//   • "file"  — drag-and-drop + file picker, with accept-filtering
//   • "photo" — a still frame from the camera (getUserMedia → <video> → canvas)
//   • "video" — a SHORT, duration-bounded webcam clip (MediaRecorder)
//   • "audio" — a SHORT, duration-bounded microphone clip (MediaRecorder)
//
// Design rules this file enforces so every demo behaves the same way (guidance ids retained in the
// build critique: privacy [rationale before powerful permission], accessibility [live regions,
// focus-visible, disabled-vs-aria-disabled, labelled controls, reduced-motion], forms [semantic
// buttons], css/css-layout [≥44px touch targets, dvh, no overflow]):
//
//   1. PERMISSION IS ALWAYS USER-INITIATED. Constructing the component NEVER calls getUserMedia.
//      A camera/mic capture only starts from an explicit click on the request button.
//   2. A PERMISSION RATIONALE is shown BEFORE the request (privacy guidance): the panel explains why
//      the demo needs the device, and the request button lives inside that panel.
//   3. EXPLICIT failure states — denied / unavailable (no device) / device-busy / unsupported
//      (no API / insecure context) — each with recovery guidance, a Retry, and the bundled fallback.
//   4. STOP / RETRY controls at every live/review step.
//   5. DURATION LIMIT + COUNTDOWN for video/audio, auto-stopping at the bound (role="timer").
//   6. DETERMINISTIC CLEANUP: every MediaStream track is track.stop()'d and every object URL is
//      URL.revokeObjectURL'd (and every ImageBitmap .close()'d) on stop, retry, error, result-consume,
//      destroy(), AND page navigation (pagehide) — no path leaks a camera light or a blob URL.
//   7. MOBILE + DESKTOP: facingMode for the camera, the `capture` attribute hint on the mobile file
//      picker, ≥44px touch targets, and a layout that never overflows horizontally.
//   8. A REQUIRED BUNDLED FALLBACK so the demo is fully usable with NO device and NO permission.
//   9. Every state transition is announced through a role="status" (polite) region, and every failure
//      through a role="alert" region.
//
// Results are emitted as { kind, source, blob, url, imageBitmap?, audioBuffer?, mime, durationMs?,
// cleanup() } — imageBitmap for "photo", audioBuffer for "audio". Call result.cleanup() when the demo
// is done with it (the component also cleans everything up on destroy()).
//
// ── Copyable usage ───────────────────────────────────────────────────────────────────────────────
//   import { createCapture, CAPTURE_CSS, injectOnce } from "/web-ai-showcase/lib/capture-ux.js";
//   injectOnce("capture-ux-css", CAPTURE_CSS);
//   const cap = createCapture({
//     mount: document.getElementById("cap"),
//     kind: "photo",                                  // "file" | "photo" | "video" | "audio"
//     accept: "image/*",                              // file kind only
//     facing: "environment",                          // camera facingMode (photo/video)
//     maxDurationMs: 8000,                            // video/audio bound
//     rationale: "Runs entirely on your device — the frame never leaves the browser.",
//     fallback: { label: "Use a sample photo", url: "/web-ai-showcase/models/foo/sample.jpg" },
//     onResult: (r) => { runInference(r.imageBitmap ?? r.blob); r.cleanup(); },
//   });
//   // later: cap.destroy();
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Inject a <style> once per document (keyed by id). Safe to call from every component instance. */
export function injectOnce(id, css) {
  if (document.getElementById(id)) return;
  const el = document.createElement("style");
  el.id = id;
  el.textContent = css;
  document.head.appendChild(el);
}

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      el.addEventListener(k.slice(2).toLowerCase(), v);
    } else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids) {
    if (kid == null) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

/** Map a getUserMedia rejection to one of our explicit states. */
function classifyGetUserMediaError(err) {
  const name = err && err.name;
  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      // SecurityError here means the permission was blocked by policy; treat as denied recovery-wise,
      // except the pre-check below routes genuine insecure-context to "unsupported".
      return { state: "denied", err };
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return { state: "unavailable", err };
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      return { state: "busy", err };
    default:
      return { state: "error", err };
  }
}

/** Pick a MediaRecorder mimeType the browser actually supports for this kind (or undefined). */
function pickRecorderMime(kind) {
  const candidates = kind === "audio"
    ? [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ]
    : [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ];
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return undefined;
  for (const c of candidates) {
    try {
      if (MediaRecorder.isTypeSupported(c)) return c;
    } catch { /* keep trying */ }
  }
  return undefined; // let the UA choose its default
}

/** Does a File / dropped item satisfy an `accept` list ("image/*,.png,audio/mpeg")? */
function matchesAccept(file, accept) {
  if (!accept || accept.trim() === "" || accept.trim() === "*") return true;
  const type = (file.type || "").toLowerCase();
  const name = (file.name || "").toLowerCase();
  return accept.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean).some((tok) => {
    if (tok.startsWith(".")) return name.endsWith(tok);
    if (tok.endsWith("/*")) return type.startsWith(tok.slice(0, -1)); // "image/" prefix
    return type === tok;
  });
}

let _decodeCtx = null;
/** A lazily-created (suspended-OK) AudioContext used only to decode captured audio to an AudioBuffer. */
function decodeCtx() {
  if (!_decodeCtx) {
    const AC = self.AudioContext || self.webkitAudioContext;
    _decodeCtx = AC ? new AC() : null;
  }
  return _decodeCtx;
}

const NEEDS_MEDIA = new Set(["photo", "video", "audio"]);

/**
 * Create a capture component mounted into `opts.mount`. Returns a controller:
 *   { el, kind, state, destroy(), reset() }
 * and calls `opts.onResult(result)` / `opts.onError(detail)` / `opts.onState(state, detail)`.
 */
export function createCapture(opts) {
  const {
    mount,
    kind = "file",
    accept = "",
    facing = "user",
    maxDurationMs = 8000,
    rationale = "",
    fallback = null, // { label, url, mime? } — REQUIRED for a device-free path; warn if missing
    captureHint = true, // add the mobile `capture` attribute hint on the file input
    onResult = () => {},
    onError = () => {},
    onState = () => {},
  } = opts;

  if (!mount) throw new Error("createCapture: opts.mount is required");
  if (!fallback && NEEDS_MEDIA.has(kind)) {
    console.warn(
      `[capture-ux] kind="${kind}" has no bundled fallback — the demo will not work without a device.`,
    );
  }

  // ── Cleanup bookkeeping (every acquired resource is tracked so destroy()/pagehide can release it) ─
  let stream = null;
  let recorder = null;
  let timerId = 0;
  let deadline = 0;
  const urls = new Set();
  const bitmaps = new Set();
  let destroyed = false;
  let state = "init";

  const trackUrl = (u) => (urls.add(u), u);
  const revoke = (u) => {
    if (u && urls.has(u)) {
      URL.revokeObjectURL(u);
      urls.delete(u);
    }
  };
  const stopStream = () => {
    if (stream) {
      for (const t of stream.getTracks()) t.stop();
      stream = null;
    }
    if (els.video) els.video.srcObject = null;
  };
  const stopRecorder = () => {
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch { /* already stopping */ }
    }
    recorder = null;
  };
  const clearTimer = () => {
    if (timerId) {
      cancelAnimationFrame(timerId);
      timerId = 0;
    }
  };

  // ── Feature / environment probes (never trigger a permission) ─────────────────────────────────────
  const hasGUM = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  const hasRecorder = typeof MediaRecorder !== "undefined";
  const secure = window.isSecureContext !== false;

  // ── DOM ───────────────────────────────────────────────────────────────────────────────────────────
  const root = h("div", { class: "cap", "data-kind": kind, "data-state": "init" });
  const els = {};

  // Shared status (polite) + error (assertive) live regions.
  els.status = h("p", { class: "cap__status", role: "status", "aria-live": "polite" });
  els.error = h("div", { class: "cap__error", role: "alert" });

  // 1) File dropzone (kind === "file")
  els.file = h("div", { class: "cap__file" });
  els.fileInput = h("input", {
    type: "file",
    class: "cap__file-input visually-hidden",
    id: `cap-file-${Math.random().toString(36).slice(2, 8)}`,
    accept: accept || null,
    // On mobile, `capture` hints the OS to offer the camera/mic directly for a media accept type.
    capture: captureHint && /image|video|audio/.test(accept) ? facing : null,
  });
  els.dropzone = h(
    "label",
    {
      class: "cap__dropzone",
      for: els.fileInput.id,
      tabindex: "0",
      role: "button",
      "aria-describedby": `${els.fileInput.id}-hint`,
    },
    h("span", { class: "cap__dz-icon", "aria-hidden": "true", html: iconUpload() }),
    h("span", {
      class: "cap__dz-text",
      html: `<strong>Choose a file</strong> or drag it here${
        accept ? ` <span class="cap__dz-accept">(${accept})</span>` : ""
      }`,
    }),
  );
  els.fileHint = h("p", {
    class: "cap__hint",
    id: `${els.fileInput.id}-hint`,
    text: accept
      ? `Accepted: ${accept}. Nothing is uploaded — the file is processed on your device.`
      : "The file is processed on your device — nothing is uploaded.",
  });
  els.file.append(els.fileInput, els.dropzone, els.fileHint);

  // 2) Permission rationale (media kinds) — shown BEFORE any getUserMedia call.
  els.rationale = h("div", { class: "cap__rationale" });
  const deviceWord = kind === "audio" ? "microphone" : "camera";
  els.rationale.append(
    h(
      "div",
      { class: "cap__rationale-head" },
      h("span", {
        class: "cap__r-icon",
        "aria-hidden": "true",
        html: kind === "audio" ? iconMic() : iconCam(),
      }),
      h(
        "p",
        { class: "cap__rationale-text" },
        rationale ||
          `This demo needs your ${deviceWord} to capture ${
            kind === "audio" ? "audio" : kind === "video" ? "a short clip" : "a photo"
          }. Access is requested only when you press the button, runs entirely on your device, and stops the moment you finish.`,
      ),
    ),
  );
  els.requestBtn = h(
    "button",
    {
      type: "button",
      class: "cap__btn cap__btn--primary",
    },
    h("span", { "aria-hidden": "true", html: kind === "audio" ? iconMic() : iconCam() }),
    h("span", {
      text: `Enable ${deviceWord}`,
    }),
  );
  els.rationale.append(els.requestBtn);

  // 3) Live stage (video preview + capture controls + countdown)
  els.stage = h("div", { class: "cap__stage" });
  els.video = h("video", {
    class: "cap__video",
    playsinline: true,
    muted: true,
    "aria-label": `${deviceWord} preview`,
  });
  els.video.muted = true; // attribute + property (some UAs need the property for autoplay of the preview)
  els.canvas = document.createElement("canvas"); // offscreen scratch for the still frame
  els.timer = h("div", { class: "cap__timer", role: "timer", "aria-live": "off", hidden: true });
  els.meter = h("div", {
    class: "cap__meter",
    role: "progressbar",
    "aria-valuemin": "0",
    "aria-valuemax": "100",
    "aria-valuenow": "0",
    "aria-label": "Recording time used",
    hidden: true,
  }, h("div", { class: "cap__meter-fill" }));
  els.stageControls = h("div", { class: "cap__controls" });
  els.stage.append(
    h("div", { class: "cap__video-wrap" }, els.video, els.timer),
    els.meter,
    els.stageControls,
  );

  // 4) Review (captured result + Use / Retake)
  els.review = h("div", { class: "cap__review" });
  els.reviewMedia = h("div", { class: "cap__review-media" });
  els.reviewControls = h("div", { class: "cap__controls" });
  els.review.append(els.reviewMedia, els.reviewControls);

  // 5) A "use the bundled sample instead" affordance, available from every media state.
  els.fallbackBar = h("div", { class: "cap__fallback" });
  if (fallback) {
    els.fallbackBtn = h(
      "button",
      {
        type: "button",
        class: "cap__btn cap__btn--ghost",
        onClick: () => useFallback(),
      },
      h("span", { "aria-hidden": "true", html: iconSample() }),
      h("span", {
        text: fallback.label || "Use a bundled sample",
      }),
    );
    els.fallbackBar.append(
      h("span", { class: "cap__fallback-lead", text: "No device? " }),
      els.fallbackBtn,
    );
  }

  root.append(
    els.error,
    els.file,
    els.rationale,
    els.stage,
    els.review,
    els.fallbackBar,
    els.status,
  );
  mount.replaceChildren(root);

  // ── State machine ─────────────────────────────────────────────────────────────────────────────────
  function setState(next, { status = "", error = "" } = {}) {
    state = next;
    root.dataset.state = next;
    ctl.state = next;
    if (status) els.status.textContent = status;
    if (error) {
      els.error.replaceChildren(...renderError(next, error));
    } else if (
      next !== "denied" && next !== "unavailable" && next !== "busy" && next !== "unsupported" &&
      next !== "error"
    ) {
      els.error.replaceChildren();
    }
    onState(next, { status, error });
  }

  function renderError(kindOfError, message) {
    const steps = recoverySteps(kindOfError);
    const nodes = [
      h(
        "div",
        { class: "cap__error-head" },
        h("span", {
          class: "cap__error-icon",
          "aria-hidden": "true",
          html: iconWarn(),
        }),
        h("strong", { text: message }),
      ),
    ];
    if (steps.length) {
      nodes.push(h("ul", { class: "cap__error-steps" }, ...steps.map((s) => h("li", { text: s }))));
    }
    const actions = h("div", { class: "cap__controls" });
    if (kindOfError !== "unsupported") {
      actions.append(
        h("button", {
          type: "button",
          class: "cap__btn",
          onClick: () => showRationale(),
        }, "Try again"),
      );
    }
    if (fallback) {
      actions.append(
        h("button", {
          type: "button",
          class: "cap__btn cap__btn--primary",
          onClick: () => useFallback(),
        }, fallback.label || "Use a bundled sample"),
      );
    }
    nodes.push(actions);
    return nodes;
  }

  function recoverySteps(kindOfError) {
    const dev = deviceWord;
    if (kindOfError === "denied") {
      return [
        `You (or the browser) blocked ${dev} access.`,
        `Click the camera/lock icon in the address bar, set ${dev} to "Allow", then press Try again.`,
        `Or continue with the bundled sample below — no permission needed.`,
      ];
    }
    if (kindOfError === "unavailable") {
      return [
        `No ${dev} was found on this device.`,
        `Connect a ${dev} (or switch to a device that has one) and press Try again.`,
        `Or continue with the bundled sample below.`,
      ];
    }
    if (kindOfError === "busy") {
      return [
        `Your ${dev} is in use by another app or tab.`,
        `Close the other app/tab using it, then press Try again.`,
      ];
    }
    if (kindOfError === "unsupported") {
      return [
        !secure
          ? "This page is not a secure context — camera/microphone access needs HTTPS (or localhost)."
          : `This browser can't access the ${dev} here.`,
        "The bundled sample below works everywhere.",
      ];
    }
    return [];
  }

  // ── Entry points ────────────────────────────────────────────────────────────────────────────────
  function showRationale() {
    stopStream();
    clearTimer();
    // Honest up-front unsupported state (still user-initiated — we just tell the truth before asking).
    if (!hasGUM || !secure || (kind !== "photo" && !hasRecorder)) {
      setState("unsupported", {
        error: !hasGUM
          ? `This browser has no ${deviceWord} API.`
          : !secure
          ? "Secure context (HTTPS) required."
          : `This browser can't record ${kind === "audio" ? "audio" : "video"} here.`,
      });
      return;
    }
    setState("idle", {
      status: `Ready. Press "Enable ${deviceWord}" when you want to grant access.`,
    });
  }

  async function requestDevice() {
    // USER-INITIATED ONLY: this runs from the request button's click handler, never on load.
    setState("requesting", {
      status:
        `Requesting ${deviceWord} access — approve the browser prompt. (Nothing has been captured yet.)`,
    });
    els.requestBtn.disabled = true;
    const constraints = kind === "audio" ? { audio: true } : {
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: kind === "video", // record sound with the webcam clip; a still photo needs no audio
    };
    try {
      const s = await navigator.mediaDevices.getUserMedia(constraints);
      if (destroyed) {
        for (const t of s.getTracks()) t.stop();
        return;
      }
      stream = s;
      if (kind !== "audio") {
        els.video.srcObject = stream;
        await els.video.play().catch(() => {});
      }
      enterPreview();
    } catch (err) {
      els.requestBtn.disabled = false;
      const { state: st } = classifyGetUserMediaError(err);
      const label = st === "denied"
        ? `${deviceWord[0].toUpperCase()}${deviceWord.slice(1)} access was blocked.`
        : st === "unavailable"
        ? `No ${deviceWord} found.`
        : st === "busy"
        ? `The ${deviceWord} is busy.`
        : `Couldn't start the ${deviceWord}: ${err.message || err.name || "unknown error"}`;
      setState(st, { error: label });
      onError({ state: st, error: err });
    }
  }

  function enterPreview() {
    els.requestBtn.disabled = false;
    els.stageControls.replaceChildren();
    if (kind === "photo") {
      els.stageControls.append(
        primaryBtn("Take photo", iconShutter(), takePhoto),
        ghostBtn("Stop camera", () => showRationale()),
      );
      setState("preview", { status: "Camera on. Frame your shot and press Take photo." });
    } else {
      els.stageControls.append(
        primaryBtn(
          kind === "audio" ? "Start recording" : "Start recording",
          iconRec(),
          startRecording,
        ),
        ghostBtn(kind === "audio" ? "Cancel" : "Stop camera", () => showRationale()),
      );
      setState("preview", {
        status: kind === "audio"
          ? "Microphone on. Press Start recording when ready."
          : "Camera on. Press Start recording when ready.",
      });
    }
  }

  // ── Photo ───────────────────────────────────────────────────────────────────────────────────────
  async function takePhoto() {
    const v = els.video;
    const w = v.videoWidth || 1280;
    const hgt = v.videoHeight || 720;
    els.canvas.width = w;
    els.canvas.height = hgt;
    const ctx = els.canvas.getContext("2d");
    ctx.drawImage(v, 0, 0, w, hgt);
    stopStream(); // release the camera the instant we have the frame
    const blob = await new Promise((res) => els.canvas.toBlob(res, "image/png"));
    let bitmap = null;
    try {
      bitmap = await createImageBitmap(els.canvas);
      bitmaps.add(bitmap);
    } catch { /* bitmap is optional */ }
    const url = trackUrl(URL.createObjectURL(blob));
    showReviewImage(url, blob, bitmap, "camera", `${w}×${hgt}`);
  }

  // ── Recording (video / audio) ─────────────────────────────────────────────────────────────────────
  function startRecording() {
    const mime = pickRecorderMime(kind);
    const chunks = [];
    try {
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    } catch (err) {
      setState("error", { error: `Recorder failed to start: ${err.message || err.name}` });
      onError({ state: "error", error: err });
      return;
    }
    recorder.addEventListener("dataavailable", (e) => {
      if (e.data && e.data.size) chunks.push(e.data);
    });
    recorder.addEventListener("error", (e) => {
      clearTimer();
      stopStream();
      setState("error", { error: `Recording error: ${e.error?.message || "unknown"}` });
      onError({ state: "error", error: e.error });
    });
    recorder.addEventListener("stop", async () => {
      clearTimer();
      const usedMime = recorder?.mimeType || mime ||
        (kind === "audio" ? "audio/webm" : "video/webm");
      stopStream();
      const blob = new Blob(chunks, { type: usedMime });
      const url = trackUrl(URL.createObjectURL(blob));
      const durationMs = maxDurationMs - Math.max(0, deadline - performance.now());
      if (kind === "audio") await showReviewAudio(url, blob, usedMime, durationMs);
      else showReviewVideo(url, blob, usedMime, durationMs);
    }, { once: true });

    recorder.start();
    // Bounded duration + countdown (guidance: role="timer", reduced-motion-safe pulse in CSS).
    deadline = performance.now() + maxDurationMs;
    els.timer.hidden = false;
    els.meter.hidden = false;
    els.stageControls.replaceChildren(
      primaryBtn("Stop recording", iconStop(), stopRecording, "cap__btn--danger"),
    );
    setState("recording", {
      status: `Recording — stops automatically after ${Math.round(maxDurationMs / 1000)} seconds.`,
    });
    tickCountdown();
  }

  function tickCountdown() {
    const loop = () => {
      if (state !== "recording") return;
      const remain = Math.max(0, deadline - performance.now());
      const secs = Math.ceil(remain / 1000);
      els.timer.textContent = `${secs}s`;
      const usedPct = Math.min(100, Math.round((1 - remain / maxDurationMs) * 100));
      els.meter.setAttribute("aria-valuenow", String(usedPct));
      els.meter.querySelector(".cap__meter-fill").style.inlineSize = `${usedPct}%`;
      if (remain <= 0) {
        stopRecording();
        return;
      }
      timerId = requestAnimationFrame(loop);
    };
    timerId = requestAnimationFrame(loop);
  }

  function stopRecording() {
    els.timer.hidden = true;
    els.meter.hidden = true;
    clearTimer();
    stopRecorder(); // fires "stop" → builds the blob + review
  }

  // ── Review renderers ──────────────────────────────────────────────────────────────────────────────
  function showReviewImage(url, blob, bitmap, source, dims) {
    els.reviewMedia.replaceChildren(
      h("img", { class: "cap__review-img", src: url, alt: "Captured photo preview" }),
    );
    const result = {
      kind: "photo",
      source,
      blob,
      url,
      imageBitmap: bitmap,
      mime: blob.type || "image/png",
      cleanup() {
        revoke(url);
        if (bitmap) {
          try {
            bitmap.close();
          } catch { /* already closed */ }
          bitmaps.delete(bitmap);
        }
      },
    };
    finishReview(result, `Captured ${dims || ""} — use it, or retake.`);
  }

  function showReviewVideo(url, blob, mime, durationMs) {
    els.reviewMedia.replaceChildren(
      h("video", { class: "cap__review-video", src: url, controls: true, playsinline: true }),
    );
    const result = {
      kind: "video",
      source: "camera",
      blob,
      url,
      mime,
      durationMs,
      cleanup: () => revoke(url),
    };
    finishReview(
      result,
      `Recorded ${(durationMs / 1000).toFixed(1)}s clip — use it, or re-record.`,
    );
  }

  async function showReviewAudio(url, blob, mime, durationMs) {
    els.reviewMedia.replaceChildren(
      h("audio", { class: "cap__review-audio", src: url, controls: true }),
    );
    let audioBuffer = null;
    try {
      const ctx = decodeCtx();
      if (ctx) audioBuffer = await ctx.decodeAudioData(await blob.arrayBuffer());
    } catch { /* decode is best-effort; blob is always present */ }
    const result = {
      kind: "audio",
      source: "mic",
      blob,
      url,
      audioBuffer,
      mime,
      durationMs,
      cleanup: () => revoke(url),
    };
    finishReview(
      result,
      `Recorded ${(durationMs / 1000).toFixed(1)}s of audio — use it, or re-record.`,
    );
  }

  function finishReview(result, status) {
    const retakeLabel = kind === "photo"
      ? "Retake"
      : kind === "file"
      ? "Choose another"
      : "Re-record";
    els.reviewControls.replaceChildren(
      primaryBtn("Use this", iconCheck(), () => {
        onResult(result);
      }),
      ghostBtn(retakeLabel, () => {
        result.cleanup();
        reset();
      }),
    );
    setState("review", { status });
  }

  // ── Fallback (bundled sample — the device-free path) ───────────────────────────────────────────────
  async function useFallback() {
    if (!fallback) return;
    setState("requesting", { status: "Loading the bundled sample…" });
    stopStream();
    clearTimer();
    try {
      const resp = await fetch(fallback.url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const raw = await resp.blob();
      const mime = fallback.mime || raw.type || guessMime(fallback.url, kind);
      const blob = raw.type ? raw : new Blob([raw], { type: mime });
      const url = trackUrl(URL.createObjectURL(blob));
      if (kind === "audio") {
        els.reviewMedia.replaceChildren(
          h("audio", { class: "cap__review-audio", src: url, controls: true }),
        );
        let audioBuffer = null;
        try {
          const ctx = decodeCtx();
          if (ctx) audioBuffer = await ctx.decodeAudioData(await blob.arrayBuffer());
        } catch { /* best-effort */ }
        finishReview({
          kind: "audio",
          source: "fallback",
          blob,
          url,
          audioBuffer,
          mime,
          cleanup: () => revoke(url),
        }, "Bundled sample loaded — use it, or pick another input.");
      } else if (kind === "video") {
        els.reviewMedia.replaceChildren(
          h("video", { class: "cap__review-video", src: url, controls: true, playsinline: true }),
        );
        finishReview({
          kind: "video",
          source: "fallback",
          blob,
          url,
          mime,
          cleanup: () => revoke(url),
        }, "Bundled sample loaded — use it, or pick another input.");
      } else {
        // photo or file → an image sample
        els.reviewMedia.replaceChildren(
          h("img", { class: "cap__review-img", src: url, alt: "Bundled sample preview" }),
        );
        let bitmap = null;
        try {
          bitmap = await createImageBitmap(blob);
          bitmaps.add(bitmap);
        } catch { /* optional */ }
        finishReview({
          kind: kind === "file" ? "file" : "photo",
          source: "fallback",
          blob,
          url,
          imageBitmap: bitmap,
          mime,
          cleanup() {
            revoke(url);
            if (bitmap) {
              try {
                bitmap.close();
              } catch { /* noop */ }
              bitmaps.delete(bitmap);
            }
          },
        }, "Bundled sample loaded — use it, or pick another input.");
      }
    } catch (err) {
      setState("error", { error: `Couldn't load the bundled sample: ${err.message}` });
      onError({ state: "error", error: err });
    }
  }

  // ── File input handling ───────────────────────────────────────────────────────────────────────────
  async function handleFile(file) {
    if (!file) return;
    if (!matchesAccept(file, accept)) {
      setState("error", {
        error: `That file type (${file.type || "unknown"}) isn't accepted here${
          accept ? `. Accepted: ${accept}` : ""
        }.`,
      });
      onError({ state: "error", error: new Error("accept-mismatch") });
      return;
    }
    const url = trackUrl(URL.createObjectURL(file));
    const isImage = (file.type || "").startsWith("image/");
    const isVideo = (file.type || "").startsWith("video/");
    const isAudio = (file.type || "").startsWith("audio/");
    if (isImage) {
      els.reviewMedia.replaceChildren(
        h("img", { class: "cap__review-img", src: url, alt: `Preview of ${file.name}` }),
      );
    } else if (isVideo) {
      els.reviewMedia.replaceChildren(
        h("video", { class: "cap__review-video", src: url, controls: true, playsinline: true }),
      );
    } else if (isAudio) {
      els.reviewMedia.replaceChildren(
        h("audio", { class: "cap__review-audio", src: url, controls: true }),
      );
    } else {
      els.reviewMedia.replaceChildren(
        h("p", {
          class: "cap__hint",
          text: `Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`,
        }),
      );
    }
    let bitmap = null;
    if (isImage) {
      try {
        bitmap = await createImageBitmap(file);
        bitmaps.add(bitmap);
      } catch { /* optional */ }
    }
    finishReview({
      kind: "file",
      source: "file",
      blob: file,
      url,
      imageBitmap: bitmap,
      mime: file.type || guessMime(file.name, "file"),
      cleanup() {
        revoke(url);
        if (bitmap) {
          try {
            bitmap.close();
          } catch { /* noop */ }
          bitmaps.delete(bitmap);
        }
      },
    }, `Selected ${file.name} — use it, or choose another.`);
  }

  // ── Small button factories ────────────────────────────────────────────────────────────────────────
  function primaryBtn(label, icon, onClick, extra = "") {
    return h(
      "button",
      { type: "button", class: `cap__btn cap__btn--primary ${extra}`, onClick },
      icon ? h("span", { "aria-hidden": "true", html: icon }) : null,
      h("span", { text: label }),
    );
  }
  function ghostBtn(label, onClick) {
    return h("button", { type: "button", class: "cap__btn cap__btn--ghost", onClick }, label);
  }

  // ── Wire up the persistent controls ───────────────────────────────────────────────────────────────
  els.requestBtn.addEventListener("click", requestDevice);
  els.fileInput.addEventListener("change", () => handleFile(els.fileInput.files[0]));
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((ev) =>
    els.dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("drag");
    })
  );
  ["dragleave", "dragend", "drop"].forEach((ev) =>
    els.dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("drag");
    })
  );
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });

  // Deterministic cleanup on navigation (pagehide fires on tab close / bfcache / SPA unload).
  const onPageHide = () => hardCleanup();
  window.addEventListener("pagehide", onPageHide);

  function hardCleanup() {
    stopRecorder();
    stopStream();
    clearTimer();
    for (const u of [...urls]) URL.revokeObjectURL(u);
    urls.clear();
    for (const b of [...bitmaps]) {
      try {
        b.close();
      } catch { /* noop */ }
    }
    bitmaps.clear();
  }

  function reset() {
    hardCleanup();
    els.reviewMedia.replaceChildren();
    els.reviewControls.replaceChildren();
    els.stageControls.replaceChildren();
    if (kind === "file") {
      els.fileInput.value = "";
      setState("idle", {
        status: accept ? `Choose or drop a file (${accept}).` : "Choose or drop a file.",
      });
    } else {
      showRationale();
    }
  }

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    window.removeEventListener("pagehide", onPageHide);
    hardCleanup();
    mount.replaceChildren();
  }

  const ctl = { el: root, kind, state, destroy, reset };

  // Initial render — NEVER requests a device. File starts at the dropzone; media starts at the
  // rationale (or an honest unsupported state).
  if (kind === "file") {
    setState("idle", {
      status: accept ? `Choose or drop a file (${accept}).` : "Choose or drop a file.",
    });
  } else {
    showRationale();
  }
  return ctl;
}

function guessMime(nameOrUrl, kind) {
  const s = String(nameOrUrl).toLowerCase();
  if (/\.png$/.test(s)) return "image/png";
  if (/\.jpe?g$/.test(s)) return "image/jpeg";
  if (/\.webp$/.test(s)) return "image/webp";
  if (/\.gif$/.test(s)) return "image/gif";
  if (/\.webm$/.test(s)) return kind === "audio" ? "audio/webm" : "video/webm";
  if (/\.mp4$/.test(s)) return "video/mp4";
  if (/\.mp3$/.test(s)) return "audio/mpeg";
  if (/\.wav$/.test(s)) return "audio/wav";
  if (/\.ogg$/.test(s)) return "audio/ogg";
  return kind === "audio" ? "audio/webm" : kind === "video" ? "video/webm" : "image/png";
}

// ── Inline SVG icons (per repo convention: inline SVG, never emoji) ──────────────────────────────────
const _svg = (p) =>
  `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
const iconCam = () =>
  _svg(
    '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
  );
const iconMic = () =>
  _svg(
    '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>',
  );
const iconUpload = () =>
  _svg(
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
  );
const iconShutter = () => _svg('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/>');
const iconRec = () => _svg('<circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>');
const iconStop = () =>
  _svg('<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>');
const iconCheck = () => _svg('<polyline points="20 6 9 17 4 12"/>');
const iconWarn = () =>
  _svg(
    '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  );
const iconSample = () =>
  _svg(
    '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
  );

/** Widget styles for the capture component — built on the shared design-system CSS variables. */
export const CAPTURE_CSS = `
.cap { display: flex; flex-direction: column; gap: .8rem; container-type: inline-size; }
.cap__status { font-family: var(--font-mono); font-size: .8rem; color: var(--muted); margin: 0; }
.cap__hint { font-size: .82rem; color: var(--muted); margin: .3rem 0 0; }

/* Buttons — ≥44px touch targets (css-layout guidance), visible focus, honest disabled state. */
.cap__btn {
  font: inherit; font-size: .92rem; display: inline-flex; align-items: center; gap: .45rem;
  min-block-size: 44px; min-inline-size: 44px; padding: .5rem .9rem; border-radius: var(--radius);
  border: 1px solid var(--border-strong); background: var(--bg-raised); color: var(--color);
  cursor: pointer; transition: background .15s, border-color .15s, opacity .15s;
}
.cap__btn:hover { border-color: var(--accent); }
.cap__btn:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.cap__btn:disabled { opacity: .5; cursor: progress; }
.cap__btn--primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.cap__btn--ghost { background: transparent; }
.cap__btn--danger { background: var(--bad); color: #fff; border-color: var(--bad); }
.cap__controls { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .6rem; }

/* File dropzone */
.cap__dropzone {
  display: flex; align-items: center; gap: .7rem; justify-content: center; text-align: center;
  border: 2px dashed var(--border-strong); border-radius: var(--radius); background: var(--bg-raised);
  padding: 1.2rem 1rem; cursor: pointer; min-block-size: 88px;
  transition: border-color .15s, background .15s;
}
.cap__dropzone.drag { border-color: var(--accent); background: var(--bg-secondary); }
.cap__dropzone:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.cap__dz-icon { color: var(--accent); display: inline-flex; }
.cap__dz-accept { color: var(--muted); font-family: var(--font-mono); font-size: .78rem; }

/* Rationale panel — shown before any getUserMedia call (privacy guidance) */
.cap__rationale {
  border: 1px solid var(--border); border-inline-start: 4px solid var(--accent);
  border-radius: var(--radius); background: var(--bg-raised); padding: 1rem;
  display: flex; flex-direction: column; gap: .7rem;
}
.cap__rationale-head { display: flex; gap: .6rem; align-items: flex-start; }
.cap__r-icon { color: var(--accent); flex: none; margin-top: .15rem; }
.cap__rationale-text { margin: 0; font-size: .95rem; }

/* Live stage */
.cap__stage { display: flex; flex-direction: column; gap: .4rem; }
.cap__video-wrap { position: relative; max-inline-size: 100%; }
.cap__video {
  inline-size: 100%; max-block-size: min(60svh, 420px); border-radius: var(--radius);
  background: #000; display: block; object-fit: contain;
}
.cap[data-kind="audio"] .cap__video-wrap { display: none; }
.cap__timer {
  position: absolute; inset-block-start: .5rem; inset-inline-end: .5rem;
  background: rgba(0,0,0,.72); color: #fff; padding: .25rem .6rem; border-radius: 999px;
  font-family: var(--font-mono); font-size: .9rem; font-variant-numeric: tabular-nums;
}
.cap__meter {
  block-size: .5rem; border-radius: 999px; overflow: hidden; border: 1px solid var(--border);
  background: var(--bg-raised); margin-top: .3rem;
}
.cap__meter-fill { block-size: 100%; inline-size: 0%; background: var(--bad); transition: inline-size .1s linear; }

/* Recording pulse on the timer — reduced-motion safe (accessibility guidance) */
.cap[data-state="recording"] .cap__timer { animation: cap-pulse 1s ease-in-out infinite; }
@keyframes cap-pulse { 50% { opacity: .5; } }
@media (prefers-reduced-motion: reduce) {
  .cap[data-state="recording"] .cap__timer { animation: none; }
  .cap__meter-fill, .cap__btn { transition: none; }
}

/* Review */
.cap__review { display: flex; flex-direction: column; gap: .3rem; }
.cap__review-img, .cap__review-video {
  inline-size: 100%; max-block-size: min(60svh, 420px); border-radius: var(--radius);
  display: block; object-fit: contain; background: var(--bg-raised); border: 1px solid var(--border);
}
.cap__review-audio { inline-size: 100%; }

/* Error / recovery */
.cap__error:empty { display: none; }
.cap__error {
  border: 1px solid var(--bad); border-radius: var(--radius); background: var(--bg-raised);
  padding: .9rem; display: flex; flex-direction: column; gap: .4rem;
}
.cap__error-head { display: flex; gap: .5rem; align-items: center; }
.cap__error-icon { color: var(--bad); display: inline-flex; flex: none; }
.cap__error-steps { margin: 0; padding-inline-start: 1.2rem; font-size: .9rem; color: var(--muted); }

/* Fallback bar */
.cap__fallback { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
.cap__fallback:empty { display: none; }
.cap__fallback-lead { color: var(--muted); font-size: .85rem; }

/* State-driven visibility — one component, many panels. */
.cap__file, .cap__rationale, .cap__stage, .cap__review { display: none; }
.cap[data-kind="file"][data-state="idle"] .cap__file { display: block; }
.cap[data-kind="file"] .cap__rationale { display: none; }
.cap[data-state="idle"]:not([data-kind="file"]) .cap__rationale,
.cap[data-state="requesting"]:not([data-kind="file"]) .cap__rationale { display: flex; }
.cap[data-state="preview"] .cap__stage,
.cap[data-state="recording"] .cap__stage { display: flex; }
.cap[data-state="review"] .cap__review { display: flex; }
/* On failure states, only the error region + fallback show. */

/* Visually-hidden native file input (label is the visible control). */
.visually-hidden:where(:not(:focus-within, :active)) {
  position: absolute !important; clip-path: inset(50%) !important; overflow: hidden !important;
  width: 1px !important; height: 1px !important; margin: -1px !important; padding: 0 !important;
  border: 0 !important; white-space: nowrap !important;
}
`;
