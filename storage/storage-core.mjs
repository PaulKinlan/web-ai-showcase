const MODEL_HOSTS = new Set([
  "huggingface.co",
  "cdn-lfs.huggingface.co",
  "cdn-lfs-us-1.huggingface.co",
  "cas-bridge.xethub.hf.co",
]);

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "Size unavailable";
  if (bytes < 1000) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = "B";
  for (const next of units) {
    value /= 1000;
    unit = next;
    if (value < 1000) break;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${unit}`;
}

export function matchesModelId(url, modelId) {
  return url.includes(`/${modelId}/`) || url.includes(encodeURIComponent(modelId));
}

export function inferHuggingFaceModelId(url) {
  try {
    const parsed = new URL(url);
    if (!MODEL_HOSTS.has(parsed.hostname)) return null;
    if (parsed.hostname !== "huggingface.co") return null;
    const match = parsed.pathname.match(/^\/(?:models\/)?([^/]+\/[^/]+)\/(?:resolve|blob)\//);
    return match ? decodeURIComponent(match[1]) : null;
  } catch {
    return null;
  }
}

function modelMeta(models, modelId) {
  const model = models.find((item) => item.hfId === modelId || item.modelId === modelId);
  return model
    ? { title: model.title || model.name || model.slug, slug: model.slug, expectedMB: model.sizeMB }
    : { title: modelId, slug: null, expectedMB: null };
}

export function buildStorageInventory(
  { records = [], cacheEntries = [], partials = [], models = [] },
) {
  const groups = new Map();
  const claimedUrls = new Set();
  for (const record of records) {
    if (!record?.modelId) continue;
    if (!groups.has(record.modelId)) {
      groups.set(record.modelId, {
        modelId: record.modelId,
        records: [],
        entries: [],
        verified: true,
        ...modelMeta(models, record.modelId),
      });
    }
    groups.get(record.modelId).records.push(record);
  }
  for (const group of groups.values()) {
    const recordUrls = new Set(group.records.flatMap((record) => record.files || []));
    group.entries = cacheEntries.filter((entry) =>
      recordUrls.has(entry.url) || matchesModelId(entry.url, group.modelId)
    );
    for (const entry of group.entries) claimedUrls.add(entry.url);
  }
  for (const entry of cacheEntries) {
    if (claimedUrls.has(entry.url)) continue;
    const modelId = inferHuggingFaceModelId(entry.url);
    if (!modelId) continue;
    if (!groups.has(modelId)) {
      groups.set(modelId, {
        modelId,
        records: [],
        entries: [],
        verified: false,
        ...modelMeta(models, modelId),
      });
    }
    groups.get(modelId).entries.push(entry);
    claimedUrls.add(entry.url);
  }
  const downloaded = [...groups.values()].map((group) => {
    const unique = [...new Map(group.entries.map((entry) => [entry.url, entry])).values()];
    const known = unique.filter((entry) => Number.isFinite(entry.bytes));
    return {
      ...group,
      entries: unique,
      fileCount: unique.length,
      knownBytes: known.reduce((sum, entry) => sum + entry.bytes, 0),
      unknownSizeFiles: unique.length - known.length,
      variants: [
        ...new Set(
          group.records.map((record) => [record.runtime, record.dtype].filter(Boolean).join(" · "))
            .filter(Boolean),
        ),
      ],
      validatedAt: group.records.map((record) =>
        record.validatedAt
      ).filter(Boolean).sort().at(-1) || null,
    };
  }).filter((group) => group.fileCount > 0 || group.records.length > 0)
    .sort((a, b) => b.knownBytes - a.knownBytes || a.title.localeCompare(b.title));
  return {
    downloaded,
    partials: partials.slice().sort((a, b) => (b.receivedBytes || 0) - (a.receivedBytes || 0)),
    claimedUrls: [...claimedUrls],
    totalKnownBytes: downloaded.reduce((sum, group) => sum + group.knownBytes, 0) +
      partials.reduce((sum, partial) => sum + (partial.receivedBytes || 0), 0),
  };
}
