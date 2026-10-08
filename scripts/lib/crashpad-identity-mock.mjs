// Pure, INERT normal-exit identity specification for web-ai-showcase-04n.
// No /proc, syscall, process spawn, signal, or production browser integration.
// Every operation is injected; ONLY fake callbacks are authorized in this stage.
// A future Python supervisor must be independently reviewed against this policy.

/** Parse /proc/PID/stat: comm (field 2) can contain spaces and parentheses. */
export function parseProcStat(text) {
  if (typeof text !== "string") return null;
  const open = text.indexOf("(");
  const close = text.lastIndexOf(")");
  if (open < 1 || close <= open || text[close + 1] !== " ") return null;
  const pidText = text.slice(0, open).trim();
  const fields = text.slice(close + 2).trim().split(/\s+/);
  if (!/^\d+$/.test(pidText) || fields.length < 20 || !/^[A-Za-z]$/.test(fields[0]) ||
      !/^\d+$/.test(fields[1]) || !/^\d+$/.test(fields[19])) return null;
  const pid = Number(pidText);
  const ppid = Number(fields[1]);
  const starttime = fields[19]; // keep the 64-bit clock-tick value lossless
  if (!Number.isSafeInteger(pid) || pid <= 1 || !Number.isSafeInteger(ppid)) return null;
  return { pid, state: fields[0], ppid, starttime };
}

/** Match a complete NUL-terminated /proc/PID/environ entry, not a substring. */
export function hasExactEnvMarker(environ, key, value) {
  if (typeof key !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ||
      typeof value !== "string" || !/^[A-Za-z0-9._-]+$/.test(value) ||
      !(typeof environ === "string" || environ instanceof Uint8Array)) return false;
  const bytes = Buffer.from(environ);
  const marker = Buffer.from(`${key}=${value}\0`);
  let offset = bytes.indexOf(marker);
  while (offset >= 0) {
    if (offset === 0 || bytes[offset - 1] === 0) return true;
    offset = bytes.indexOf(marker, offset + 1);
  }
  return false;
}

/** Discovery records identity BEFORE opening a pidfd. No signal is possible here. */
export function discoverCandidate(pid, expectedPpid, readProc) {
  if (!Number.isSafeInteger(pid) || pid <= 1 || !Number.isSafeInteger(expectedPpid) ||
      expectedPpid <= 1 || typeof readProc !== "function") return null;
  try {
    const stat = parseProcStat(readProc(pid, "stat"));
    if (!stat || stat.pid !== pid || stat.ppid !== expectedPpid || stat.state === "Z") return null;
    return Object.freeze({ pid, ppid: expectedPpid, starttime: stat.starttime });
  } catch { return null; }
}

/**
 * Synthetic fd-call sequence; do NOT pass real syscalls or host /proc here.
 * pidfdOpen pins first; post-pin identity must equal discovery identity before
 * any pidfdSignal(fd, sig). Missing/changed evidence skips, never a PID signal.
 */
export function bindAndSignalMock(candidate, io, { markerKey, markerValue, sig = "SIGTERM" } = {}) {
  const skipped = (reason) => ({ action: "skipped", reason });
  if (!candidate || !Number.isSafeInteger(candidate.pid) || candidate.pid <= 1 ||
      !Number.isSafeInteger(candidate.ppid) || candidate.ppid <= 1 ||
      typeof candidate.starttime !== "string" || !/^\d+$/.test(candidate.starttime) ||
      typeof io?.pidfdOpen !== "function" || typeof io.readProc !== "function" ||
      typeof io.pidfdSignal !== "function" || typeof io.pidfdClose !== "function" ||
      typeof markerKey !== "string" || typeof markerValue !== "string")
    return skipped("invalid candidate, marker, or injected operations");
  let fd;
  let result;
  try {
    fd = io.pidfdOpen(candidate.pid);
    if (!Number.isSafeInteger(fd) || fd < 0) {
      result = skipped("pidfd_open did not return a valid fd");
    } else {
      const stat = parseProcStat(io.readProc(candidate.pid, "stat"));
      if (!stat || stat.pid !== candidate.pid || stat.ppid !== candidate.ppid ||
          stat.starttime !== candidate.starttime || stat.state === "Z") {
        result = skipped("post-pin stat differs from discovered identity or is unavailable");
      } else if (!hasExactEnvMarker(io.readProc(candidate.pid, "environ"), markerKey, markerValue)) {
        result = skipped("post-pin exact environment marker is missing or unavailable");
      } else {
        io.pidfdSignal(fd, sig); // fd ONLY; never pass a numeric PID to a signal callback
        result = { action: "signaled" };
      }
    }
  } catch {
    result = skipped("injected open/read/signal operation failed");
  } finally {
    if (Number.isSafeInteger(fd) && fd >= 0) {
      try { io.pidfdClose(fd); }
      catch {
        // A close failure after signal cannot undo a signal; report it honestly.
        result = { ...(result || skipped("no decision")), closeFailed: true };
      }
    }
  }
  return result;
}
