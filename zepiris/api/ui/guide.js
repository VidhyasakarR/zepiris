// Half-body capture guide shared by /ui and /ui/selfie.
//
// Front camera + face-oval / T-shirt-to-stomach outline + a live MediaPipe
// skeleton. There is no auto-capture: the outline turns green when the pose is
// aligned, and only then is capture allowed (onReady(true)). If the pose model
// cannot load (slow CDN, no WebGL) capture is allowed anyway, so a rider is
// never stuck.
//
// The stage can be any size (a 3:4 card, or the whole phone screen). The camera
// picture is laid out inside it by a zoom factor:
//
//   zoom 1     the picture covers the stage (fills it; the sides get cropped —
//              on a tall phone screen that is effectively zoomed in)
//   zoom < 1   the picture is shown smaller, revealing more of the camera's real
//              field of view, down to the whole frame (never smaller than that)
//
// A web page cannot widen the lens itself; zooming out here means cropping less.
// The guide, the skeleton and the saved photo all use the VISIBLE part of the
// camera picture (the "content" rectangle), so they always agree.
//
// Geometry is in content units: 100 wide × Hc tall (Hc = 100 · h/w of the
// visible picture), drawn without stretching. Vertical positions scale with
// k = Hc / 133.33, so at 3:4 (k = 1) the rules are exactly the ones calibrated
// on real half-body rider selfies (nose ~y43, shoulders ~y78, shoulder width
// ~65) and match the Android app's pose_guide.dart.

const MP_VERSION = "0.10.14";
const MP_BUNDLE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`;
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MP_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const MP_FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

// 1 = fill the screen (crops the camera picture on a tall phone); lower = show
// more of it. Below the "whole picture" point (about 0.6 on a 9:20 phone with a
// 3:4 camera) the picture no longer grows smaller: it is all shown, with bars.
export const ZOOM_LEVELS = [0.25, 0.5, 0.75, 0.9, 1];
const clampZoom = (z) => Math.min(1, Math.max(0.25, Number(z) || 1));

/** Guide geometry for a content area 100 wide × Hc tall. */
export function geometry(Hc) {
  const k = Hc / 133.33;
  const oval = { cx: 50, cy: 38 * k, rx: 18, ry: 23 };
  const neck = oval.cy + oval.ry; // where the torso outline starts
  const Y = (off) => Math.min(Hc, neck + off * k);
  return {
    Hc, k, oval, neck,
    // the torso outline, offsets from the neck scaled by k (shape of the 3:4 original)
    torso: `M50 ${Y(0)} C 44 ${Y(1)} 42 ${Y(5)} 40 ${Y(7)} C 30 ${Y(9)} 18 ${Y(12)} 14 ${Y(20)} C 11 ${Y(28)} 10 ${Y(45)} 10 ${Hc} L 90 ${Hc} C 90 ${Y(45)} 89 ${Y(28)} 86 ${Y(20)} C 82 ${Y(12)} 70 ${Y(9)} 60 ${Y(7)} C 58 ${Y(5)} 56 ${Y(1)} 50 ${Y(0)} Z`,
    edges: `M40 ${Y(7)} C 30 ${Y(9)} 18 ${Y(12)} 14 ${Y(20)} C 11 ${Y(28)} 10 ${Y(45)} 10 ${Hc} M60 ${Y(7)} C 70 ${Y(9)} 82 ${Y(12)} 86 ${Y(20)} C 89 ${Y(28)} 90 ${Y(45)} 90 ${Hc}`,
    stomachY: Y(63),
    labels: { face: Math.max(4, oval.cy - oval.ry - 4), shirt: Y(43), stomach: Math.min(Hc - 2, Y(69)) },
    // rule bands (at k = 1: shoulders 64–92, hips must be below 110)
    shoulderMin: Y(3), shoulderMax: Y(31), hipMin: Y(49),
  };
}

function guideSvg(g) {
  const { Hc, oval: o } = g;
  return `
    <defs>
      <mask id="guide-cut">
        <rect width="100" height="${Hc}" fill="#fff"/>
        <ellipse cx="${o.cx}" cy="${o.cy}" rx="${o.rx}" ry="${o.ry}" fill="#000"/>
        <path d="${g.torso}" fill="#000"/>
      </mask>
    </defs>
    <rect class="guide-dim" width="100" height="${Hc}" mask="url(#guide-cut)"/>
    <ellipse class="guide-line" cx="${o.cx}" cy="${o.cy}" rx="${o.rx}" ry="${o.ry}"/>
    <path class="guide-line" d="${g.edges}"/>
    <line class="guide-line" x1="16" y1="${g.stomachY}" x2="84" y2="${g.stomachY}" opacity=".6"/>
    <text class="guide-label" x="50" y="${g.labels.face}">FACE IN OVAL</text>
    <text class="guide-label" x="50" y="${g.labels.shirt}">T-SHIRT + LOGO</text>
    <text class="guide-label" x="50" y="${g.labels.stomach}">STOMACH LINE</text>`;
}

const seen = (p, min = 0.5) => p && p.v > min;

/** Alignment rules for geometry `g` → null when aligned, else the instruction to show. */
export function assess(pts, g = geometry(133.33)) {
  const { oval: o } = g;
  const n = pts[0], le = pts[7], re = pts[8], ls = pts[11], rs = pts[12], lh = pts[23], rh = pts[24];
  if (!seen(n)) return "Show your face to the camera";
  // Nose well inside the oval (0.7 of it), so the face sits in it, not on its edge.
  const d = ((n.x - o.cx) / o.rx) ** 2 + ((n.y - o.cy) / o.ry) ** 2;
  if (d > 0.7) return n.y < o.cy - o.ry * 0.6 ? "Lower the phone — face in the oval"
    : n.y > o.cy + o.ry * 0.6 ? "Raise the phone — face in the oval" : "Centre your face in the oval";
  if (seen(le) && seen(re)) {
    // Ear-to-ear is ~3/4 of face width; the oval is 36 units wide.
    const earW = Math.abs(le.x - re.x);
    if (earW < 14) return "Come closer — fill the oval";
    if (earW > 44) return "Move the phone back";
  }
  if (!seen(ls) || !seen(rs)) return "Step back — show both shoulders";
  const shW = Math.abs(ls.x - rs.x), shY = (ls.y + rs.y) / 2;
  if (shW > 88) return "Step back — too close";
  if (shW < 42) return "Come closer";
  if (shY < g.shoulderMin) return "Step back — show the T-shirt";
  if (shY > g.shoulderMax) return "Raise the phone a little";
  const hipY = seen(lh, 0.3) && seen(rh, 0.3) ? (lh.y + rh.y) / 2 : Infinity;
  if (hipY < g.hipMin) return "Step back — show the T-shirt down to the stomach";
  return null;
}

/** SVG for the live skeleton: head outline, neck, shoulders, arms, torso. */
function skeletonSvg(pts) {
  const P = (i, min = 0.4) => (seen(pts[i], min) ? pts[i] : null);
  const n = P(0), le = P(7), re = P(8), ls = P(11), rs = P(12);
  let s = "";
  const line = (a, b, cls = "") => { if (a && b) s += `<line class="${cls}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`; };
  let head = null;
  if (n) {
    const earW = le && re ? Math.abs(le.x - re.x) : null;
    const rx = earW ? earW * 0.68 : 11;
    const cx = le && re ? (le.x + re.x) / 2 : n.x;
    const cy = n.y - rx * 0.25;
    head = { cx, cy, rx, ry: rx * 1.3 };
    s += `<ellipse class="head" cx="${cx}" cy="${cy}" rx="${head.rx}" ry="${head.ry}"/>`;
  }
  if (head && ls && rs) line({ x: head.cx, y: head.cy + head.ry }, { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 }, "neck");
  line(ls, rs);
  line(ls, P(13)); line(P(13), P(15));
  line(rs, P(14)); line(P(14), P(16));
  line(ls, P(23, 0.3)); line(rs, P(24, 0.3)); line(P(23, 0.3), P(24, 0.3));
  for (const i of [11, 12, 13, 14, 15, 16]) { const p = P(i); if (p) s += `<circle cx="${p.x}" cy="${p.y}" r="1.1"/>`; }
  for (const i of [23, 24]) { const p = P(i, 0.3); if (p) s += `<circle cx="${p.x}" cy="${p.y}" r="1.1"/>`; }
  return s;
}

// ---- blur check -------------------------------------------------------------
// Crété-Roffet no-reference blur metric (2007) on the face: re-blur the image and
// measure how much edge detail survives. 0 = sharp … 1 = fully blurred. Unlike a
// raw Laplacian variance it barely depends on lighting or skin/background
// texture. Calibrated IN THE BROWSER with this exact function (canvas resampling
// shifts the scale vs. OpenCV) on the 20 real rider photos: genuine selfies
// score 0.31–0.48; the one tiny-face full-body shot scores 0.80. At 0.65 no real
// selfie is flagged, heavy blur (σ≈4) is caught 89% of the time, and on a live
// capture the warning fires just before the logo check starts failing (logo
// still passed at 0.69, failed at 0.74) — the logo is the first check to go.
export const BLUR_THRESHOLD = 0.65;
const BLUR_WIDTH = 160;

/** Blur score (0 sharp … 1 blurred) of `rect` = {x, y, w, h} (pixels) of `canvas`. */
export function blurScore(canvas, rect) {
  const w = BLUR_WIDTH, h = Math.max(8, Math.round((rect.h * w) / rect.w));
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingQuality = "high";
  g.drawImage(canvas, rect.x, rect.y, rect.w, rect.h, 0, 0, w, h);
  const px = g.getImageData(0, 0, w, h).data;
  const F = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) F[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  const R = 4; // 9-tap box blur
  const along = (horizontal) => {
    const B = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let sum = 0, n = 0;
      for (let d = -R; d <= R; d++) {
        const xx = horizontal ? x + d : x, yy = horizontal ? y : y + d;
        if (xx >= 0 && xx < w && yy >= 0 && yy < h) { sum += F[yy * w + xx]; n++; }
      }
      B[y * w + x] = sum / n;
    }
    let sF = 0, sV = 0;
    for (let y = horizontal ? 0 : 1; y < h; y++) for (let x = horizontal ? 1 : 0; x < w; x++) {
      const i = y * w + x, j = horizontal ? i - 1 : i - w;
      const dF = Math.abs(F[i] - F[j]), dB = Math.abs(B[i] - B[j]);
      sF += dF; sV += Math.max(0, dF - dB);
    }
    return sF > 0 ? (sF - sV) / sF : 1;
  };
  return Math.max(along(true), along(false));
}

// ---- low light --------------------------------------------------------------
// Face brightness is mean luma (0–255) of the face box. Indoor-lit faces sit
// around 100–170; below LOW_LIGHT the camera stretches exposure (noise + motion
// smear), so Capture lights the face with a white "screen flash" first. After
// capture a face under BRIGHTEN_BELOW is gamma-lifted toward BRIGHTEN_TO before
// it is sent; one under TOO_DARK even then is flagged — too little signal for
// the checks (and the blur metric reads sensor noise as detail, so it cannot be
// trusted there either).
export const LOW_LIGHT = 90, BRIGHTEN_BELOW = 110, BRIGHTEN_TO = 128, TOO_DARK = 45;
// Overexposed face (detail clipped), and a face much darker than a bright scene
// behind it (window / lamp behind the rider).
export const TOO_BRIGHT = 225, BACKLIT_SCENE = 140, BACKLIT_RATIO = 0.6;
// Screen flash: keep the screen white while the camera's auto-exposure catches
// up, sampling the face every FLASH_STEP_MS and keeping the brightest frame.
// Webcams often wait 0.2–0.4 s before exposure moves at all, then ramp over
// ~1 s, so: at least FLASH_MIN_MS, then stop once no sample has beaten the best
// by 2% for FLASH_SETTLE_MS, and never longer than FLASH_MAX_MS.
const FLASH_MIN_MS = 800, FLASH_SETTLE_MS = 400, FLASH_MAX_MS = 1800, FLASH_STEP_MS = 100;
// Burst: motion blur varies a lot frame to frame (a hand shakes in bursts), so
// Capture takes several frames and keeps the sharpest face.
const BURST_FRAMES = 5, BURST_STEP_MS = 70;

/**
 * Server-side capture quality (POST /v1/quality/check): the ResNet blur model on
 * the face and the T-shirt, plus visibility and light. Resolves to the API's
 * {ok, warnings:[{code, region, message}], face, shirt}, or null if unreachable.
 */
export const QUALITY_TIMEOUT_MS = 6000;
export async function serverQuality(dataUrl, apiBase = "", timeoutMs = QUALITY_TIMEOUT_MS) {
  // A stalled network must not leave the rider on "Checking the photo…": give
  // up after timeoutMs and fall back to the in-browser reading.
  const ctl = typeof AbortController === "function" ? new AbortController() : null;
  const timer = ctl && setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(`${apiBase}/v1/quality/check`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_b64: dataUrl }),
      signal: ctl?.signal,
    });
    const d = r.ok ? await r.json() : null;
    return d && Array.isArray(d.warnings) ? d : null;
  } catch (_) {
    return null;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Mean and std-dev of luma over `rect` of `source` (a canvas or the video). */
export function faceLuma(source, rect) {
  const w = 64, h = Math.max(8, Math.round((rect.h * w) / rect.w));
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(source, rect.x, rect.y, rect.w, rect.h, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h).data;
  let sum = 0, sq = 0; const n = w * h;
  for (let i = 0; i < n; i++) { const y = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]; sum += y; sq += y * y; }
  const mean = sum / n;
  return { mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)) };
}

/** Gamma-lift a canvas in place so a face at `mean` lands near BRIGHTEN_TO. */
function brighten(canvas, mean) {
  const gamma = Math.min(1, Math.max(0.35, Math.log(BRIGHTEN_TO / 255) / Math.log(Math.max(1, mean) / 255)));
  const lut = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) lut[i] = 255 * (i / 255) ** gamma;
  const g = canvas.getContext("2d", { willReadFrequently: true });
  const img = g.getImageData(0, 0, canvas.width, canvas.height), d = img.data;
  for (let i = 0; i < d.length; i += 4) { d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]]; }
  g.putImageData(img, 0, 0);
  return gamma;
}

/** A full-viewport white overlay: the phone screen lights the rider's face. */
function screenFlash() {
  const el = document.createElement("div");
  el.className = "flash-screen";
  el.style.cssText = "position:fixed;inset:0;background:#fff;z-index:2147483647;opacity:1;transition:opacity .25s";
  document.body.append(el);
  return () => { el.style.opacity = "0"; setTimeout(() => el.remove(), 260); };
}

/** The face region of the captured photo, from the last aligned landmarks (content units). */
function faceRect(pts, cw, ch) {
  const u = cw / 100; // content units → captured-photo pixels (same scale on both axes)
  const n = pts && seen(pts[0], 0.4) ? pts[0] : null;
  if (!n) return { x: cw * 0.3, y: ch * 0.12, w: cw * 0.4, h: cw * 0.4 }; // no pose: upper-centre
  const le = seen(pts[7], 0.4) ? pts[7] : null, re = seen(pts[8], 0.4) ? pts[8] : null;
  const earW = le && re ? Math.abs(le.x - re.x) : 20;
  const side = Math.max(earW * 1.36, 16) * 1.2 * u;           // head width, padded 1.2×
  const cx = (le && re ? (le.x + re.x) / 2 : n.x) * u, cy = n.y * u - earW * 0.17 * u;
  const x = Math.max(0, cx - side / 2), y = Math.max(0, cy - side / 2);
  return { x, y, w: Math.min(cw - x, side), h: Math.min(ch - y, side) };
}

// ---- live face analysis (MediaPipe Face Landmarker: 478 points + blendshapes) --
// Drives the pre-capture checks that make a photo usable before it is taken —
// one face, eyes open, looking straight, holding still — and the active
// liveness challenge (blink / turn), which a printed photo or a replayed still
// cannot perform. The challenge runs in the browser, so it is a UX barrier, not
// proof: the server-side passive liveness model still decides at the checkpoint.
export const EYES_CLOSED = 0.5, EYES_OPEN = 0.3;   // blendshape eyeBlink scores
// Calibrated on the 18 real rider selfies: frontal yaw within ±0.08, pitch
// (re-centred) within ±0.05, eyeBlink ≤ 0.25 with eyes open.
export const YAW_FRONTAL = 0.11, YAW_TURNED = 0.18; // nose offset from face centre, fraction of face width
export const PITCH_FRONTAL = 0.12;
export const STILL_PX = 0.012, STILL_MS = 450;      // nose travel (fraction of frame width) allowed over STILL_MS
const FACE_LOST_RESET_MS = 1200;                     // face gone this long → the challenge must be redone

/** Per-frame face metrics from a FaceLandmarker result (null when no face). */
export function faceMetrics(res) {
  const f = res?.faceLandmarks?.[0];
  if (!f) return null;
  const bs = Object.fromEntries((res.faceBlendshapes?.[0]?.categories || []).map((c) => [c.categoryName, c.score]));
  const l = bs.eyeBlinkLeft ?? 0, r = bs.eyeBlinkRight ?? 0;
  const L = f[234], R = f[454], N = f[1], T = f[10], B = f[152]; // cheeks, nose tip, forehead, chin
  return {
    count: res.faceLandmarks.length,
    eyesClosed: Math.min(l, r),    // both eyes shut (a blink)
    eyeClosed: Math.max(l, r),     // at least one eye shut
    yaw: (N.x - L.x) / Math.max(1e-6, R.x - L.x) - 0.5,
    pitch: (N.y - T.y) / Math.max(1e-6, B.y - T.y) - 0.54,
    nose: { x: N.x, y: N.y },
  };
}

// The WebAssembly runtime, fetched and compiled once for both models.
let filesPromise = null;
function visionFiles() {
  filesPromise = filesPromise || (async () => {
    const { FilesetResolver } = await import(MP_BUNDLE);
    return FilesetResolver.forVisionTasks(MP_WASM);
  })();
  filesPromise.catch(() => { filesPromise = null; });
  return filesPromise;
}

// A model setup that never settles (a WebView put in the background mid-load
// can stall GPU setup indefinitely) must not hang the camera screen: give each
// attempt a deadline, then fall back / let the next start() retry.
const GPU_SETUP_MS = 10000, CPU_SETUP_MS = 20000;
function withDeadline(p, ms, what) {
  let t;
  return Promise.race([p, new Promise((_, rej) => { t = setTimeout(() => rej(new Error(`${what} timed out`)), ms); })])
    .finally(() => clearTimeout(t));
}

// Run one frame through a fresh model: the first inference compiles the GPU
// shaders (up to a second on a phone), better paid before the camera opens.
function warmUp(model) {
  try {
    const c = document.createElement("canvas"); c.width = c.height = 128;
    const g = c.getContext("2d"); g.fillStyle = "#888"; g.fillRect(0, 0, 128, 128);
    model.detectForVideo(c, performance.now());
  } catch (_) { /* warm-up is best effort */ }
  return model;
}

let faceLmPromise = null;
function loadFaceLandmarker() {
  // A failed load is forgotten, so a retake / Retry tries the network again.
  faceLmPromise = faceLmPromise || (async () => {
    const [{ FaceLandmarker }, files] = await Promise.all([import(MP_BUNDLE), visionFiles()]);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MP_FACE_MODEL, delegate }, runningMode: "VIDEO",
      numFaces: 2, outputFaceBlendshapes: true,
    });
    try { return warmUp(await withDeadline(FaceLandmarker.createFromOptions(files, opts("GPU")), GPU_SETUP_MS, "face GPU")); }
    catch (_) { return warmUp(await withDeadline(FaceLandmarker.createFromOptions(files, opts("CPU")), CPU_SETUP_MS, "face CPU")); }
  })();
  faceLmPromise.catch(() => { faceLmPromise = null; });
  return faceLmPromise;
}

/**
 * What is wrong with the live face right now, in priority order; null = all good.
 * `face` is faceMetrics() output (null = no face), `challenge` a createChallenge(),
 * `noseTrail` recent nose positions [{x, y}], `light` {tooBright, backlit}.
 */
export function liveFaceProblem(face, { light = {}, challenge, noseTrail = [] }) {
  if (!face) return "Look at the camera";
  if (face.count > 1) return "Only one person in the frame";
  if (light.tooBright) return "Too bright — move out of direct light";
  if (light.backlit) return "Light is behind you — face the light";
  // during the turn challenge turning is the point: don't ask to face front
  const turning = challenge.type === "turn" && !challenge.done;
  if (Math.abs(face.yaw) > YAW_FRONTAL && !turning) return "Look straight at the camera";
  if (Math.abs(face.pitch) > PITCH_FRONTAL) return "Keep your head level";
  if (challenge.type === "blink" && !challenge.done) return null; // blinking is expected here
  if (face.eyeClosed > EYES_CLOSED) return "Open your eyes";
  const moving = noseTrail.length > 1 && Math.max(...noseTrail.map((p) => Math.hypot(p.x - face.nose.x, p.y - face.nose.y))) > STILL_PX;
  if (moving && challenge.done) return "Hold still";
  return null;
}

const CHALLENGE_TEXT = { blink: "Blink your eyes", turn: "Turn your head to one side, then back" };

/**
 * Active liveness challenge state machine. `step(face)` takes one frame's
 * faceMetrics and returns true once the challenge is complete:
 *   blink: both eyes seen closed (> EYES_CLOSED), then open again (< EYES_OPEN)
 *   turn : head turned past YAW_TURNED either way, then back to frontal
 */
// Blink, measured against this rider's own open-eye level: a blink behind
// glasses or in dim light often peaks at 0.35-0.45 on the blendshape, which a
// fixed 0.5 missed ("blink again, and again"). Closed = both eyes rise at least
// BLINK_RISE over the open baseline (and past BLINK_MIN_CLOSED); open again =
// back within BLINK_REOPEN of it. A still photo never rises; a wink raises only
// one eye (the less-closed eye is what is compared).
export const BLINK_RISE = 0.2, BLINK_MIN_CLOSED = 0.3, BLINK_REOPEN = 0.1;

export function createChallenge(type) {
  const c = { type, phase: 0, done: type === "none", base: null };
  c.reset = () => { c.phase = 0; c.done = c.type === "none"; c.base = null; };
  c.closedAt = () => Math.min(EYES_CLOSED, Math.max(BLINK_MIN_CLOSED, (c.base ?? 0) + BLINK_RISE));
  c.step = (face) => {
    if (c.done || !face) return c.done;
    if (c.type === "blink") {
      const both = face.eyesClosed, either = face.eyeClosed;
      if (c.base === null) c.base = both;
      if (c.phase === 0) {
        if (both >= c.closedAt()) c.phase = 1;
        else c.base = 0.85 * c.base + 0.15 * both;   // follow the open level (lighting, squint)
      } else if (either <= Math.max(EYES_OPEN, c.base + BLINK_REOPEN)) {
        c.done = true;
      }
    } else if (c.type === "turn") {
      if (c.phase === 0 && Math.abs(face.yaw) > YAW_TURNED) c.phase = 1;
      else if (c.phase === 1 && Math.abs(face.yaw) < YAW_FRONTAL * 0.7) c.done = true;
    }
    return c.done;
  };
  return c;
}

let landmarkerPromise = null;
function loadLandmarker() {
  // One model per page, shared across retakes.
  landmarkerPromise = landmarkerPromise || (async () => {
    const [{ PoseLandmarker }, files] = await Promise.all([import(MP_BUNDLE), visionFiles()]);
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: MP_MODEL, delegate }, runningMode: "VIDEO", numPoses: 1 });
    try { return warmUp(await withDeadline(PoseLandmarker.createFromOptions(files, opts("GPU")), GPU_SETUP_MS, "pose GPU")); }
    catch (_) { return warmUp(await withDeadline(PoseLandmarker.createFromOptions(files, opts("CPU")), CPU_SETUP_MS, "pose CPU")); }
  })();
  landmarkerPromise.catch(() => { landmarkerPromise = null; });
  return landmarkerPromise;
}

/**
 * Download, compile and warm up both camera models ahead of time (e.g. while
 * the app shows its config screen). createCapture().start() then finds them
 * ready instead of spending seconds on "Loading the camera guide…".
 * Resolves true when both loaded.
 */
export async function preload() {
  const r = await Promise.allSettled([loadLandmarker(), loadFaceLandmarker()]);
  return r.every((x) => x.status === "fulfilled");
}

/**
 * Mount the guide into `stage` (an empty element with class "stage").
 *
 * options:
 *   zoom                 initial zoom (0.5–1, default 1); chips let the user change it
 *   zoomLevels           chips to show (default [0.75, 0.8, 0.9, 1]); [] hides them
 *   onReady(canCapture)  capture allowed? true while aligned (or when the pose
 *                        model is unavailable); drive the Capture button with it
 *   onCapture(dataUrl, quality)  JPEG data URL of the visible picture (≤ maxSide px)
 *                        and {blur, blurry}: the face's blur score and whether it
 *                        is over BLUR_THRESHOLD (warn the rider to retake)
 *   onState(state, err)  "live" | "captured" | "error"
 *   onPoseStatus(text)   pose-model status line
 *   onLight(low)         the face is in low light (the screen will flash on capture)
 *   flash                "auto" (flash in low light, default) | "on" | "off"
 *   challenge            active liveness before capture: "none" (default) | "blink" |
 *                        "turn" | "random"
 *   onChallenge(state)   {type, done} whenever the challenge changes
 *   maxSide              default 1600 (API caps images at 5 MB)
 */
export function createCapture(stage, options = {}) {
  const {
    onReady = () => {}, onCapture = () => {}, onState = () => {}, onPoseStatus = () => {},
    maxSide = 1600, zoomLevels = ZOOM_LEVELS, onLight = () => {}, flash: flashMode = "auto",
    onChallenge = () => {},
  } = options;
  const pickChallenge = (t) => (t === "random" ? (Math.random() < 0.5 ? "blink" : "turn") : t);
  const challengeType = options.challenge === "random"
    ? pickChallenge("random")
    : ["blink", "turn"].includes(options.challenge) ? options.challenge : "none";
  let zoom = clampZoom(options.zoom ?? 1);

  stage.innerHTML = `
    <div class="content">
      <svg class="guide" aria-hidden="true"></svg>
      <svg class="skeleton" aria-hidden="true"></svg>
    </div>
    <div class="idle"><p>Fit your face in the oval<br>and your T-shirt down to the stomach.</p></div>
    <div class="flash"></div>
    <div class="light">💡 Low light — the screen will flash to light your face</div>
    <div class="zoom" hidden>${zoomLevels.map((z) => `<button type="button" data-z="${z}">${z === 1 ? "1" : z}×</button>`).join("")}</div>
    <div class="banner">Camera off</div>`;
  const q = (s) => stage.querySelector(s);
  const content = q(".content"), guide = q(".guide"), skeleton = q(".skeleton");
  const banner = q(".banner"), idle = q(".idle"), flash = q(".flash"), zoomBar = q(".zoom");

  let stream = null, video = null, landmarker = null, poseFailed = false, rafId = 0, lastTs = -1;
  let ready = false, geo = null, lay = null, lastPts = null;
  let lowLight = false, lastLightTs = 0, capturing = false, alignedTs = 0, starting = null, pausedByHide = false;
  let gen = 0; // bumped by stop(): a start() that was overtaken gives up after its next await
  let faceLm = null, faceFailed = false, face = null, faceSeenTs = 0, tick = 0, poseLm = null;
  let light = { tooBright: false, backlit: false };
  const noseTrail = []; // {t, x, y} over the last STILL_MS
  const challenge = createChallenge(challengeType);
  const resetChallenge = () => {
    challenge.reset();
    onChallenge({ type: challenge.type, done: challenge.done });
  };
  // Once aligned, Capture stays enabled this long through a missed frame, so a
  // noisy low-light feed or a shaky hand does not make the button flicker.
  const READY_HOLD_MS = 400;

  const setReady = (v) => { if (v !== ready) { ready = v; onReady(v); } };
  const setBanner = (text, aligned = false) => { banner.textContent = text; stage.classList.toggle("aligned", aligned); };

  // ---- layout: where the camera picture sits, and what part of it is visible --
  function layout() {
    const sW = stage.clientWidth, sH = stage.clientHeight;
    if (!sW || !sH) return;
    const vw = video?.videoWidth || 3, vh = video?.videoHeight || 4; // idle: assume a 3:4 camera
    const cover = Math.max(sW / vw, sH / vh), contain = Math.min(sW / vw, sH / vh);
    const scale = Math.max(contain, cover * zoom);
    const dW = vw * scale, dH = vh * scale, left = (sW - dW) / 2, top = (sH - dH) / 2;
    // visible content = the displayed picture clipped to the stage
    const cx = Math.max(0, left), cy = Math.max(0, top), cw = Math.min(sW, dW), ch = Math.min(sH, dH);
    lay = { sW, sH, scale, dW, dH, left, top, cx, cy, cw, ch };
    if (video) Object.assign(video.style, { left: `${left}px`, top: `${top}px`, width: `${dW}px`, height: `${dH}px` });
    Object.assign(content.style, { left: `${cx}px`, top: `${cy}px`, width: `${cw}px`, height: `${ch}px` });
    const Hc = +(100 * ch / cw).toFixed(2);
    if (!geo || Math.abs(geo.Hc - Hc) > 0.25) {
      geo = geometry(Hc);
      for (const svg of [guide, skeleton]) svg.setAttribute("viewBox", `0 0 100 ${Hc}`);
      guide.innerHTML = guideSvg(geo);
    }
    zoomBar.querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.z === zoom));
  }
  new ResizeObserver(() => layout()).observe(stage);
  layout();

  // App switched away / screen locked: Android and iOS stop or freeze the camera.
  // Release it while hidden and reopen it on return (unless a photo was taken).
  let hostHidden = false; // an app hosting this page in a WebView says it is paused
  const hidden = () => document.hidden || hostHidden;
  function onVisibility() {
    if (hidden()) {
      // mid-start counts too: stop() makes the in-flight start() give up
      if ((stream || starting) && !capturing) { pausedByHide = true; stop(); setBanner("Camera paused"); }
    } else if (pausedByHide) {
      pausedByHide = false;
      // a start() that was in flight when hidden has given up: start afresh after it
      Promise.resolve(starting).then(() => { if (!stream && !hidden() && !stage.classList.contains("captured")) start(); });
    }
  }
  document.addEventListener("visibilitychange", onVisibility);
  // Android WebViews keep running (and keep the camera) when the app goes to the
  // background; the host app sends this instead.
  window.addEventListener("zepiris-host-visibility", (e) => { hostHidden = !!e.detail?.hidden; onVisibility(); });

  zoomBar.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-z]");
    if (b) setZoom(+b.dataset.z);
  });
  function setZoom(z) { zoom = clampZoom(z); layout(); }

  // A full-frame landmark (0–1) → content units (100 × Hc).
  function toContent(p) {
    const px = lay.left + p.x * lay.dW, py = lay.top + p.y * lay.dH;
    return { x: ((px - lay.cx) / lay.cw) * 100, y: ((py - lay.cy) / lay.cw) * 100, v: p.visibility ?? 1 };
  }

  // Pre-capture checks on the live face (see liveFaceProblem). null = all good.
  function faceProblem() {
    if (faceFailed) return null; // model unavailable: framing checks only
    if (face && face.count > 1 && challenge.type !== "none" && (challenge.done || challenge.phase)) {
      resetChallenge(); // a second person: the challenge is not handed over
    }
    return liveFaceProblem(face, { light, challenge, noseTrail });
  }


  function stepChallenge() {
    if (challenge.done || !face) return;
    if (challenge.step(face)) { noseTrail.length = 0; onChallenge({ type: challenge.type, done: true }); }
  }

  function loop() {
    if (!stream || !video) return;
    rafId = requestAnimationFrame(loop);
    if (!landmarker || video.readyState < 2 || !lay || capturing) return; // capturing: keep Capture locked
    const ts = performance.now();
    // While the challenge runs the face model gets the frames: a blink lasts
    // ~150 ms, so it needs ~25 face readings a second; pose runs every 3rd.
    const pending = !challenge.done;
    if (ts - lastTs < (pending ? 30 : 50)) return;
    lastTs = ts;
    tick++;

    if (faceLm) {
      face = faceMetrics(faceLm.detectForVideo(video, ts));
      if (face) {
        faceSeenTs = ts;
        noseTrail.push({ t: ts, ...face.nose });
        while (noseTrail.length && ts - noseTrail[0].t > STILL_MS) noseTrail.shift();
        // Count the challenge on every good face frame, not only when the body
        // framing is also perfect: a blink during a framing flicker still counts.
        if (pending && face.count === 1 && (challenge.type === "turn"
            || (Math.abs(face.yaw) <= YAW_FRONTAL * 1.5 && Math.abs(face.pitch) <= PITCH_FRONTAL * 1.5))) {
          stepChallenge();
        }
      } else if (ts - faceSeenTs > FACE_LOST_RESET_MS && challenge.type !== "none" && (challenge.done || challenge.phase)) {
        resetChallenge(); // a different person could step in after the challenge
      }
    }
    if (tick % (pending ? 3 : 2) === 1 || !poseLm) poseLm = landmarker.detectForVideo(video, ts).landmarks?.[0] || null;

    if (!poseLm) {
      skeleton.innerHTML = "";
      if (ts - alignedTs > READY_HOLD_MS) { setBanner("Step into the frame"); setReady(false); }
      return;
    }
    const pts = poseLm.map(toContent);
    lastPts = pts;
    skeleton.innerHTML = skeletonSvg(pts);
    if (ts - lastLightTs > 400) { lastLightTs = ts; checkLight(); }

    const problem = assess(pts, geo) || faceProblem();
    if (!problem) {
      if (!challenge.done) {
        // framed well: now the liveness challenge, Capture still locked
        setBanner(`${CHALLENGE_TEXT[challenge.type]}${challenge.phase ? " …" : ""}`, false);
        stage.classList.add("challenge");
        setReady(false);
        return;
      }
      stage.classList.remove("challenge");
      alignedTs = ts;
      setBanner("Perfect — tap Capture", true);
      setReady(true);
    } else if (ts - alignedTs > READY_HOLD_MS) {
      stage.classList.remove("challenge");
      setBanner(problem, false);
      setReady(false);
    }
  }

  // The visible part of the camera picture, in camera pixels.
  const visibleSrc = () => ({
    sx: (lay.cx - lay.left) / lay.scale, sy: (lay.cy - lay.top) / lay.scale,
    sw: lay.cw / lay.scale, sh: lay.ch / lay.scale,
  });

  function checkLight() {
    try {
      const v = visibleSrc(), r = faceRect(lastPts, v.sw, v.sh);
      const faceMean = faceLuma(video, { x: v.sx + r.x, y: v.sy + r.y, w: r.w, h: r.h }).mean;
      const frameMean = faceLuma(video, { x: v.sx, y: v.sy, w: v.sw, h: v.sh }).mean;
      light = {
        tooBright: faceMean > TOO_BRIGHT,
        // a bright window/lamp behind: the scene is lit, the face is not
        backlit: frameMean > BACKLIT_SCENE && faceMean < frameMean * BACKLIT_RATIO,
      };
      const low = faceMean < LOW_LIGHT;
      if (low !== lowLight) { lowLight = low; stage.classList.toggle("lowlight", low); onLight(low); }
    } catch (_) { /* a skipped reading is harmless */ }
  }

  // Concurrent start() calls (double tap, visibility + retake) share one attempt.
  function start() {
    if (!starting) starting = doStart().finally(() => { starting = null; });
    return starting;
  }

  async function doStart() {
    stop();
    const my = gen, stale = () => my !== gen;
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        // Asked in the SENSOR's (landscape) orientation, with no resizing: Chrome on
        // Android reads width/height that way, and a portrait request (1080×1440)
        // made it crop the frame to a narrow strip — a heavily zoomed-in picture.
        // This returns the camera's native 3:4 mode (1080×1440 portrait on a phone),
        // the full field of view the phone's own camera app shows.
        video: { facingMode: "user", width: { ideal: 1440 }, height: { ideal: 1080 }, resizeMode: "none" },
        audio: false,
      });
      if (stale()) { s.getTracks().forEach((t) => t.stop()); return false; }
      stream = s;
    } catch (err) {
      if (stale()) return false;
      setBanner(!window.isSecureContext ? "Camera needs https or localhost"
        : err.name === "NotAllowedError" ? "Camera permission denied — allow it in the browser"
        : err.name === "NotFoundError" ? "No camera found" : "Camera unavailable");
      onState("error", err);
      return false;
    }
    stage.querySelectorAll("video, img.shot").forEach((n) => n.remove());
    stage.classList.remove("captured", "blurry", "lowlight", "challenge");
    lastPts = null; lowLight = false; capturing = false; face = null; poseLm = null; noseTrail.length = 0;
    light = { tooBright: false, backlit: false };
    idle.hidden = true;
    video = document.createElement("video");
    Object.assign(video, { autoplay: true, playsInline: true, muted: true, srcObject: stream });
    video.setAttribute("playsinline", ""); // iOS Safari/WKWebView: inline, not full screen
    stage.prepend(video);
    await new Promise((r) => (video.readyState >= 1 ? r() : (video.onloadedmetadata = r)));
    if (stale()) return false;
    try { await video.play(); } catch (_) { /* autoplay covers it where play() is refused */ }
    if (stale()) return false;
    // The camera can change resolution (rotation, another app, a driver switch):
    // re-fit the picture and the outline when it does.
    video.addEventListener("resize", layout);
    const track = stream.getVideoTracks()[0];
    // Where the camera itself can zoom out (wide front lenses report zoom < 1),
    // use its widest setting: a real wider view, not just less cropping.
    try {
      const caps = track?.getCapabilities?.();
      if (caps?.zoom && caps.zoom.min < (track.getSettings?.().zoom ?? 1)) {
        await track.applyConstraints({ advanced: [{ zoom: caps.zoom.min }] });
      }
    } catch (_) { /* not supported: layout zoom only */ }
    if (stale()) return false;
    if (track) track.onended = () => {
      if (!stream || capturing) return;
      stop();
      setBanner("Camera stopped — tap Start camera");
      onState("error", new Error("camera track ended"));
    };
    layout();
    zoomBar.hidden = zoomLevels.length === 0;
    setBanner("Fit your face in the oval");
    onState("live");
    if ((!landmarker && !poseFailed) || (!faceLm && !faceFailed)) {
      onPoseStatus("Loading the camera guide…");
      const [pose, fl] = await Promise.allSettled([
        landmarker || poseFailed ? Promise.resolve(landmarker) : loadLandmarker(),
        faceLm || faceFailed ? Promise.resolve(faceLm) : loadFaceLandmarker(),
      ]);
      if (stale()) return false;
      if (pose.status === "fulfilled") landmarker = pose.value; else poseFailed = true;
      if (fl.status === "fulfilled") faceLm = fl.value; else faceFailed = true;
      onPoseStatus(poseFailed ? "Pose guide unavailable — frame yourself in the outline and tap Capture"
        : faceFailed ? "Face checks unavailable — framing guide only" : "");
    }
    if (challenge.type !== "none" && (poseFailed || faceFailed)) {
      // Fail closed: the liveness challenge can't run without the models, and
      // skipping it would let a photo through. Offer a retry instead.
      poseFailed = faceFailed = false; // the next start() loads them again
      stop();
      setBanner("Couldn't load the face check — check the connection and tap Start camera");
      onPoseStatus("");
      onState("error", new Error("face models unavailable"));
      return false;
    }
    resetChallenge();
    if (poseFailed) setReady(true); // no challenge asked: framing guide only
    loop();
    return true;
  }

  function stop() {
    gen++;
    cancelAnimationFrame(rafId);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    skeleton.innerHTML = "";
    stage.classList.remove("aligned");
    zoomBar.hidden = true;
    setReady(false);
  }

  async function capture() {
    if (!video || !video.videoWidth || !ready || !lay || capturing) return null;
    capturing = true;
    setReady(false);
    zoomBar.hidden = true; // a zoom change mid-burst would crop frames differently
    try {
      return await takeShot();
    } catch (err) {
      // Never leave the rider stuck on a white screen with a dead Capture button.
      document.querySelectorAll(".flash-screen").forEach((n) => n.remove());
      setBanner("Capture failed — try again");
      if (stream) { zoomBar.hidden = zoomLevels.length === 0; alignedTs = 0; }
      return null;
    } finally {
      capturing = false;
    }
  }

  async function takeShot() {
    // Low light: light the face with the screen, give auto-exposure a moment, and
    // take the frame while the screen is still white.
    const useFlash = flashMode === "on" || (flashMode === "auto" && lowLight);
    // Exactly the visible part of the picture, in camera pixels, unmirrored.
    const { sx, sy, sw, sh } = visibleSrc();
    const out = Math.min(1, maxSide / Math.max(sw, sh));
    const grab = () => {
      const k = document.createElement("canvas");
      k.width = Math.round(sw * out); k.height = Math.round(sh * out);
      k.getContext("2d").drawImage(video, sx, sy, sw, sh, 0, 0, k.width, k.height);
      return k;
    };
    const frames = []; // {canvas, mean}
    const sample = () => { const k = grab(); frames.push({ canvas: k, mean: faceLuma(k, faceRect(lastPts, k.width, k.height)).mean }); };
    // Face brightness straight off the video (a 64 px read): the flash ramp is
    // watched this way, so it doesn't hold ~20 full-size canvases in memory
    // (iOS WebViews refuse new canvases past their memory cap).
    const r0 = faceRect(lastPts, sw, sh);
    const liveMean = () => faceLuma(video, { x: sx + r0.x, y: sy + r0.y, w: r0.w, h: r0.h }).mean;
    if (useFlash) {
      const endFlash = screenFlash();
      const t0 = performance.now();
      let bestMean = -1, bestTs = t0;
      while (performance.now() - t0 < FLASH_MAX_MS) {
        await new Promise((r) => setTimeout(r, FLASH_STEP_MS));
        const m = liveMean(), now = performance.now();
        if (m > bestMean * 1.02) bestTs = now;
        bestMean = Math.max(bestMean, m);
        // exposure has settled: past the minimum and no real gain for a while
        if (now - t0 >= FLASH_MIN_MS && now - bestTs >= FLASH_SETTLE_MS) break;
      }
      try {
        for (let i = 0; i < 3; i++) { await new Promise((r) => setTimeout(r, BURST_STEP_MS)); sample(); } // settled frames
      } finally {
        endFlash();
      }
    } else {
      sample();
      for (let i = 1; i < BURST_FRAMES; i++) { await new Promise((r) => setTimeout(r, BURST_STEP_MS)); sample(); }
    }
    // Among the well-lit frames (≥ 90% of the brightest), keep the sharpest face.
    const top = Math.max(...frames.map((f) => f.mean));
    let c = frames[frames.length - 1].canvas, bestBlur = Infinity;
    for (const f of frames.filter((f) => f.mean >= top * 0.9)) {
      try {
        const b = blurScore(f.canvas, faceRect(lastPts, f.canvas.width, f.canvas.height));
        if (b < bestBlur) { bestBlur = b; c = f.canvas; }
      } catch (_) { /* keep the current pick */ }
    }

    for (const f of frames) if (f.canvas !== c) f.canvas.width = 0; // free the unused frames now
    const quality = {
      blur: null, blurry: false, brightness: null, brightened: false, dark: false, flashed: useFlash,
      challenge: { type: challenge.type, passed: challenge.done, ran: !faceFailed },
    };
    try {
      const face = faceRect(lastPts, c.width, c.height);
      const { mean } = faceLuma(c, face);
      quality.brightness = Math.round(mean);
      quality.dark = mean < TOO_DARK;
      // Blur on the photo as taken (gamma does not change the metric much) — but
      // in the dark it reads noise as detail, so a dark photo is never "sharp".
      const blur = blurScore(c, face);
      quality.blur = +blur.toFixed(3);
      quality.blurry = !quality.dark && blur > BLUR_THRESHOLD;
      if (mean < BRIGHTEN_BELOW) {
        // The server light check must see the photo as the camera took it: after
        // brightening every dark photo would look fine to it.
        quality.raw = c.toDataURL("image/jpeg", 0.9);
        brighten(c, mean); quality.brightened = true;
      }
    } catch (_) { /* never block a capture on the checks themselves */ }
    const dataUrl = c.toDataURL("image/jpeg", 0.92);

    flash.classList.remove("go"); void flash.offsetWidth; flash.classList.add("go");
    stop();
    video.remove(); video = null;
    const shot = new Image(); shot.className = "shot"; shot.src = dataUrl; shot.alt = "";
    Object.assign(shot.style, { left: `${lay.cx}px`, top: `${lay.cy}px`, width: `${lay.cw}px`, height: `${lay.ch}px` });
    stage.prepend(shot);
    stage.classList.add("captured"); // show the photo itself, not the outline over it
    const bad = quality.dark || quality.blurry;
    setBanner(quality.dark ? "Too dark — please retake" : quality.blurry ? "Blurry — please retake" : "Captured");
    stage.classList.toggle("blurry", bad);
    onState("captured");
    onCapture(dataUrl, quality);
    return dataUrl;
  }

  return {
    start, stop, capture, setZoom,
    /** Change the liveness challenge ("none" | "blink" | "turn" | "random") before start(). */
    setChallenge(t) {
      const p = pickChallenge(t || "blink");
      challenge.type = ["blink", "turn", "none"].includes(p) ? p : "blink";
      resetChallenge();
    },
    get zoom() { return zoom; }, get live() { return !!stream; }, get ready() { return ready; },
  };
}
