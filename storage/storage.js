import {
  clearCachedUrls,
  clearModelRecords,
  listValidationRecords,
  scanCacheInventory,
  storageEstimate,
} from "../lib/model-cache.js";
import { clearPartial, listResumeStates } from "../lib/model-download.js";
import { buildStorageInventory, formatBytes, inferHuggingFaceModelId } from "./storage-core.mjs";

const els = {
  status: document.querySelector("#status"),
  downloaded: document.querySelector("#downloaded"),
  partialSection: document.querySelector("#partial-section"),
  partials: document.querySelector("#partials"),
  modelCount: document.querySelector("#model-count"),
  knownSize: document.querySelector("#known-size"),
  originUsage: document.querySelector("#origin-usage"),
  quotaCopy: document.querySelector("#quota-copy"),
  quotaMeter: document.querySelector("#quota-meter"),
  refresh: document.querySelector("#refresh"),
  clearAll: document.querySelector("#clear-all"),
  dialog: document.querySelector("#confirm-dialog"),
  dialogTitle: document.querySelector("#confirm-title"),
  dialogCopy: document.querySelector("#confirm-copy"),
  cancel: document.querySelector("#cancel-delete"),
  confirm: document.querySelector("#confirm-delete"),
};

let inventory = { downloaded: [], partials: [], totalKnownBytes: 0 };
let pendingAction = null;
let returnFocus = null;

function empty(message) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = message;
  return p;
}

function detailLine(group) {
  const parts = [];
  if (group.knownBytes) parts.push(formatBytes(group.knownBytes));
  else if (group.expectedMB) parts.push(`Approximately ${group.expectedMB} MB when complete`);
  else parts.push("Size unavailable");
  parts.push(`${group.fileCount} cached file${group.fileCount === 1 ? "" : "s"}`);
  if (group.unknownSizeFiles) parts.push(`${group.unknownSizeFiles} without reported byte sizes`);
  return parts.join(" · ");
}

function downloadedItem(group, index) {
  const li = document.createElement("li");
  li.className = "storage-item";
  const body = document.createElement("div");
  const title = document.createElement("h3");
  if (group.slug) {
    const link = document.createElement("a");
    link.href = `../models/${group.slug}/`;
    link.textContent = group.title;
    title.append(link);
  } else title.textContent = group.title;
  const id = document.createElement("p");
  id.className = "muted";
  const code = document.createElement("code");
  code.textContent = group.modelId;
  id.append(code);
  const details = document.createElement("p");
  details.textContent = detailLine(group);
  const state = document.createElement("p");
  state.className = "muted";
  state.textContent = group.verified
    ? `Validated local model${group.variants.length ? ` · ${group.variants.join(" / ")}` : ""}${
      group.validatedAt ? ` · ${new Date(group.validatedAt).toLocaleString()}` : ""
    }`
    : "Cached model files that have not completed validation.";
  body.append(title, id, details, state);
  const actions = document.createElement("div");
  actions.className = "item-actions";
  const remove = document.createElement("button");
  remove.className = "danger";
  remove.type = "button";
  remove.textContent = "Delete model";
  remove.dataset.modelIndex = String(index);
  actions.append(remove);
  li.append(body, actions);
  return li;
}

function partialItem(partial, index) {
  const li = document.createElement("li");
  li.className = "storage-item";
  const body = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = inferHuggingFaceModelId(partial.url) || "Interrupted model download";
  const path = document.createElement("p");
  path.className = "muted";
  const code = document.createElement("code");
  code.textContent = partial.url;
  path.append(code);
  const details = document.createElement("p");
  const total = partial.total ? ` of ${formatBytes(partial.total)}` : "";
  details.textContent = `${formatBytes(partial.receivedBytes || 0)} saved${total}${
    partial.updatedAt ? ` · ${new Date(partial.updatedAt).toLocaleString()}` : ""
  }`;
  body.append(title, path, details);
  const remove = document.createElement("button");
  remove.className = "danger";
  remove.type = "button";
  remove.textContent = "Delete partial download";
  remove.dataset.partialIndex = String(index);
  li.append(body, remove);
  return li;
}

function render(estimate) {
  els.downloaded.replaceChildren();
  if (!inventory.downloaded.length) {
    els.downloaded.append(empty("No downloaded model files were found in this browser profile."));
  } else {
    const list = document.createElement("ul");
    list.className = "storage-list";
    inventory.downloaded.forEach((group, index) => list.append(downloadedItem(group, index)));
    els.downloaded.append(list);
  }
  els.partials.replaceChildren();
  els.partialSection.hidden = inventory.partials.length === 0;
  if (inventory.partials.length) {
    const list = document.createElement("ul");
    list.className = "storage-list";
    inventory.partials.forEach((partial, index) => list.append(partialItem(partial, index)));
    els.partials.append(list);
  }
  els.modelCount.textContent = String(inventory.downloaded.length);
  els.knownSize.textContent = inventory.totalKnownBytes
    ? `At least ${formatBytes(inventory.totalKnownBytes)}`
    : "None reported";
  if (estimate && Number.isFinite(estimate.usage)) {
    els.originUsage.textContent = formatBytes(estimate.usage);
    if (Number.isFinite(estimate.quota) && estimate.quota > 0) {
      const ratio = Math.min(1, estimate.usage / estimate.quota);
      els.quotaCopy.textContent = `${formatBytes(estimate.usage)} used of ${
        formatBytes(estimate.quota)
      } available to this site. Total includes app files and browser overhead.`;
      els.quotaMeter.hidden = false;
      els.quotaMeter.value = ratio;
      els.quotaMeter.querySelector("span").textContent = `${Math.round(ratio * 100)}%`;
    }
  } else {
    els.originUsage.textContent = "Unavailable";
    els.quotaCopy.textContent = "This browser does not expose a storage estimate.";
    els.quotaMeter.hidden = true;
  }
  els.clearAll.disabled = inventory.downloaded.length === 0 && inventory.partials.length === 0;
}

async function refresh(message = "") {
  els.refresh.disabled = true;
  els.status.textContent = "Checking local model storage…";
  try {
    const [records, cacheEntries, partials, modelsResponse, estimate] = await Promise.all([
      listValidationRecords(),
      scanCacheInventory(),
      listResumeStates(),
      fetch("../models.json").then((response) => response.ok ? response.json() : { models: [] }),
      storageEstimate(),
    ]);
    inventory = buildStorageInventory({
      records,
      cacheEntries,
      partials,
      models: modelsResponse.models || [],
    });
    render(estimate);
    els.status.textContent = message ||
      `Found ${inventory.downloaded.length} downloaded model${
        inventory.downloaded.length === 1 ? "" : "s"
      }${
        inventory.partials.length
          ? ` and ${inventory.partials.length} interrupted download${
            inventory.partials.length === 1 ? "" : "s"
          }`
          : ""
      }.`;
  } catch (error) {
    console.error(error);
    els.status.textContent = "Could not inspect local model storage in this browser.";
    els.downloaded.replaceChildren(
      empty(
        "Local storage could not be read. Private browsing or browser policy may block access.",
      ),
    );
  } finally {
    els.refresh.disabled = false;
  }
}

function askDelete({ title, copy, action }, invoker) {
  pendingAction = action;
  returnFocus = invoker;
  els.dialogTitle.textContent = title;
  els.dialogCopy.textContent = copy;
  els.confirm.textContent = title.startsWith("Delete all") ? "Delete all model data" : "Delete";
  if (typeof els.dialog.showModal === "function") els.dialog.showModal();
  else if (window.confirm(copy)) runPendingAction();
}

async function runPendingAction() {
  if (!pendingAction) return;
  const action = pendingAction;
  pendingAction = null;
  els.confirm.disabled = true;
  els.cancel.disabled = true;
  try {
    const message = await action();
    if (els.dialog.open) els.dialog.close();
    await refresh(message);
  } catch (error) {
    console.error(error);
    els.dialogCopy.textContent =
      "Deletion failed. The browser may be blocking storage access. Nothing else was removed.";
  } finally {
    els.confirm.disabled = false;
    els.cancel.disabled = false;
  }
}

els.downloaded.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-model-index]");
  if (!button) return;
  const group = inventory.downloaded[Number(button.dataset.modelIndex)];
  if (!group) return;
  askDelete({
    title: "Delete model?",
    copy: `Delete ${group.title} from this browser? Its demos will need to download it again.`,
    action: async () => {
      const files = await clearCachedUrls(group.entries.map((entry) => entry.url));
      await clearModelRecords(group.modelId);
      return `Deleted ${group.title} (${files} cached file${files === 1 ? "" : "s"}).`;
    },
  }, button);
});

els.partials.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-partial-index]");
  if (!button) return;
  const partial = inventory.partials[Number(button.dataset.partialIndex)];
  if (!partial) return;
  askDelete({
    title: "Delete partial download?",
    copy: `Delete ${
      formatBytes(partial.receivedBytes || 0)
    } of resumable download data? A future download will restart from zero.`,
    action: async () => {
      await clearPartial(partial.url);
      return "Deleted the interrupted download.";
    },
  }, button);
});

els.clearAll.addEventListener("click", () =>
  askDelete({
    title: "Delete all model data?",
    copy: `Delete ${inventory.downloaded.length} downloaded model${
      inventory.downloaded.length === 1 ? "" : "s"
    } and ${inventory.partials.length} interrupted download${
      inventory.partials.length === 1 ? "" : "s"
    }? App files and unrelated browser data will be kept.`,
    action: async () => {
      const files = await clearCachedUrls(
        inventory.downloaded.flatMap((group) => group.entries.map((entry) => entry.url)),
      );
      for (const group of inventory.downloaded) await clearModelRecords(group.modelId);
      for (const partial of inventory.partials) await clearPartial(partial.url);
      return `Deleted all listed model data (${files} cached file${files === 1 ? "" : "s"}).`;
    },
  }, els.clearAll));

els.refresh.addEventListener("click", () => refresh());
els.cancel.addEventListener("click", () => els.dialog.close());
els.confirm.addEventListener("click", runPendingAction);
els.dialog.addEventListener("close", () => {
  pendingAction = null;
  returnFocus?.focus();
  returnFocus = null;
});

refresh();
