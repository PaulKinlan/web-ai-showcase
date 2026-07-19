// Perceptual colour maps for the Depth Anything pages, shared by BOTH the main thread (legend +
// backward-compatible renderDepthColor) and the inference worker (the off-main-thread colourise
// composite). Keeping the ramp + the colourise loop here means the colourised depth map is
// byte-identical wherever it is produced — main-thread render and worker composite can never drift.
//
// Pure data + math, no DOM — so it imports cleanly into a module Worker.

// Control-point stops per map, interpolated in sRGB.
const MAPS = {
  turbo: [
    [48, 18, 59],
    [65, 69, 171],
    [57, 118, 209],
    [32, 163, 181],
    [48, 196, 120],
    [140, 208, 52],
    [216, 182, 29],
    [238, 116, 32],
    [165, 20, 24],
  ],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma: [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  gray: [[0, 0, 0], [255, 255, 255]],
};

export const COLORMAPS = Object.keys(MAPS);

/** Sample a named colour map at t∈[0,1] → [r,g,b]. */
export function sampleMap(name, t) {
  const stops = MAPS[name] ?? MAPS.turbo;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(stops.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/**
 * Colourise a single-channel depth buffer (0–255) into a flat RGBA Uint8ClampedArray (w*h*4).
 * This is the dense per-pixel composite — run it inside the worker (off the main thread) and transfer
 * the resulting ImageBitmap back, so the main thread never pays the ~O(pixels) cost at 1080p+.
 */
export function colorizeDepth(depth, width, height, mapName = "turbo") {
  const px = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < depth.length; i++) {
    const [r, g, b] = sampleMap(mapName, depth[i] / 255);
    const o = i * 4;
    px[o] = r;
    px[o + 1] = g;
    px[o + 2] = b;
    px[o + 3] = 255;
  }
  return px;
}

export { MAPS };
