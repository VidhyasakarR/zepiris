// Unit tests for the pure parts of zepiris/api/ui/guide.js (no browser needed).
// Run: node tests/js/guide.test.mjs   (also run by tests/test_ui_guide_js.py)
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const src = readFileSync(new URL("../../zepiris/api/ui/guide.js", import.meta.url), "utf8");
const G = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));
let passed = 0;
const test = (name, fn) => { fn(); passed++; };

// ---- framing rules (stage units, 3:4) -------------------------------------
const pose = ({ nose, shoulderY, shoulderW, hipY, ears }) => {
  const p = [];
  p[0] = { x: nose[0], y: nose[1], v: 1 };
  if (ears) { p[7] = { x: nose[0] + ears / 2, y: nose[1], v: 1 }; p[8] = { x: nose[0] - ears / 2, y: nose[1], v: 1 }; }
  p[11] = { x: 50 + shoulderW / 2, y: shoulderY, v: 1 };
  p[12] = { x: 50 - shoulderW / 2, y: shoulderY, v: 1 };
  if (hipY) { p[23] = { x: 60, y: hipY, v: 1 }; p[24] = { x: 40, y: hipY, v: 1 }; }
  return p;
};
const g = G.geometry(133.33);

test("real half-body selfies are aligned", () => {
  // measured on the rider photos: p2 nose (61,41) shoulders y78 w66; p1 nose (52,46) y79 w64
  assert.equal(G.assess(pose({ nose: [55, 41], shoulderY: 78, shoulderW: 66, hipY: 169, ears: 24 }), g), null);
  assert.equal(G.assess(pose({ nose: [52, 42], shoulderY: 79, shoulderW: 64, hipY: 158, ears: 22 }), g), null);
});
test("framing problems give the right instruction", () => {
  assert.equal(G.assess([], g), "Show your face to the camera");
  assert.equal(G.assess(pose({ nose: [19, 27], shoulderY: 40, shoulderW: 20 }), g), "Centre your face in the oval");
  assert.equal(G.assess(pose({ nose: [50, 38], shoulderY: 80, shoulderW: 95 }), g), "Step back — too close");
  assert.equal(G.assess(pose({ nose: [50, 38], shoulderY: 80, shoulderW: 30 }), g), "Come closer");
  assert.equal(G.assess(pose({ nose: [50, 38], shoulderY: 80, shoulderW: 60, ears: 8 }), g), "Come closer — fill the oval");
  assert.equal(G.assess(pose({ nose: [50, 38], shoulderY: 80, shoulderW: 60, hipY: 100 }), g), "Step back — show the T-shirt down to the stomach");
});
test("geometry scales with the visible picture's shape", () => {
  const tall = G.geometry(177.8); // 9:16
  assert.ok(tall.oval.cy > g.oval.cy && tall.oval.rx === g.oval.rx, "oval moves down, keeps its shape");
  assert.ok(tall.shoulderMax > g.shoulderMax);
});

// ---- face metrics -----------------------------------------------------------
const faceRes = ({ yaw = 0, blinkL = 0.1, blinkR = 0.1, faces = 1, openL = null, openR = null }) => {
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  // eyelid outline (only when a test sets openness 0..1): width 0.04, open height 0.012
  const eye = (open, cx, [up1, lo1, up2, lo2, c1, c2]) => {
    const h = 0.012 * open;
    lm[c1] = { x: cx - 0.02, y: 0.4 }; lm[c2] = { x: cx + 0.02, y: 0.4 };
    lm[up1] = { x: cx, y: 0.4 - h / 2 }; lm[lo1] = { x: cx, y: 0.4 + h / 2 };
    lm[up2] = { x: cx + 0.008, y: 0.4 - h / 2 }; lm[lo2] = { x: cx + 0.008, y: 0.4 + h / 2 };
  };
  if (openL != null) eye(openL, 0.46, [159, 145, 158, 153, 33, 133]);
  if (openR != null) eye(openR, 0.54, [386, 374, 385, 380, 362, 263]);
  lm[234] = { x: 0.40, y: 0.45 }; lm[454] = { x: 0.60, y: 0.45 };   // cheeks: face 0.2 wide
  lm[1] = { x: 0.50 + yaw * 0.2, y: 0.46 };                            // nose tip
  lm[10] = { x: 0.5, y: 0.30 }; lm[152] = { x: 0.5, y: 0.60 };         // forehead, chin
  return {
    faceLandmarks: Array.from({ length: faces }, () => lm),
    faceBlendshapes: [{ categories: [{ categoryName: "eyeBlinkLeft", score: blinkL }, { categoryName: "eyeBlinkRight", score: blinkR }] }],
  };
};
test("faceMetrics reads yaw, eyes and face count", () => {
  const m = G.faceMetrics(faceRes({ yaw: 0.2, blinkL: 0.9, blinkR: 0.2, faces: 2 }));
  assert.ok(Math.abs(m.yaw - 0.2) < 1e-9);
  assert.equal(m.count, 2);
  assert.equal(m.eyesClosed, 0.2); // both eyes: the less-closed one
  assert.equal(m.eyeClosed, 0.9);  // either eye
  assert.equal(G.faceMetrics({ faceLandmarks: [] }), null);
});

// ---- liveness challenge ---------------------------------------------------
const run = (type, frames) => { const c = G.createChallenge(type); frames.forEach((f) => c.step(G.faceMetrics(faceRes(f)))); return c.done; };
const open = { blinkL: 0.1, blinkR: 0.1 }, shut = { blinkL: 0.8, blinkR: 0.75 };
test("blink: open → closed → open passes", () => assert.equal(run("blink", [open, open, shut, shut, open]), true));
test("blink: a still photo never passes", () => assert.equal(run("blink", Array(50).fill(open)), false));
test("blink: a one-eye wink does not count", () => assert.equal(run("blink", [open, { blinkL: 0.9, blinkR: 0.1 }, open]), false));
test("blink: eyes that never reopen do not pass", () => assert.equal(run("blink", [open, shut, shut, shut]), false));
test("turn: frontal → turned → frontal passes", () => assert.equal(run("turn", [{ yaw: 0 }, { yaw: 0.25 }, { yaw: 0.02 }]), true));
test("turn: a small head wobble does not pass", () => assert.equal(run("turn", [{ yaw: 0 }, { yaw: 0.1 }, { yaw: 0 }]), false));
test("none: passes immediately", () => assert.equal(G.createChallenge("none").done, true));
test("reset makes it start over", () => {
  const c = G.createChallenge("blink");
  [open, shut, open].forEach((f) => c.step(G.faceMetrics(faceRes(f))));
  assert.equal(c.done, true); c.reset(); assert.equal(c.done, false);
});

// ---- blink measured against the rider's own open-eye level ------------------
const eyes = (l, r = l) => ({ blinkL: l, blinkR: r });
test("blink: a soft blink behind glasses (peak 0.4) passes", () =>
  assert.equal(run("blink", [eyes(0.08), eyes(0.1), eyes(0.09), eyes(0.4), eyes(0.12)]), true));
test("blink: squinting rider (open level 0.3) blinks to 0.55", () =>
  assert.equal(run("blink", [eyes(0.3), eyes(0.28), eyes(0.31), eyes(0.55), eyes(0.33)]), true));
test("blink: jittery open eyes (0.05-0.25) never pass", () => {
  const f = []; for (let i = 0; i < 200; i++) f.push(eyes(0.05 + 0.2 * ((i * 37) % 11) / 10));
  assert.equal(run("blink", f), false);
});
test("blink: one frame of it is enough (fast blink)", () =>
  assert.equal(run("blink", [eyes(0.1), eyes(0.6), eyes(0.1)]), true));
test("blink: a wink still does not count", () =>
  assert.equal(run("blink", [eyes(0.1), eyes(0.9, 0.12), eyes(0.1), eyes(0.1, 0.85), eyes(0.1)]), false));

// ---- eyelid signal: works when the blink score stays flat (glasses) --------
const lids = (l, r = l) => ({ blinkL: 0.1, blinkR: 0.1, openL: l, openR: r });
test("eyelids: a blink with a flat blink score (glasses) passes", () =>
  assert.equal(run("blink", [lids(1), lids(1), lids(0.95), lids(0.2), lids(0.95)]), true));
test("eyelids: one closed frame of a fast blink is enough", () =>
  assert.equal(run("blink", [lids(1), lids(0.4), lids(1)]), true));
test("eyelids: a still photo never passes", () => assert.equal(run("blink", Array(60).fill(lids(1))), false));
test("eyelids: jitter of ±15% never passes", () => {
  const f = []; for (let i = 0; i < 200; i++) f.push(lids(0.85 + 0.3 * ((i * 37) % 11) / 10, 0.85 + 0.3 * ((i * 53) % 7) / 6));
  assert.equal(run("blink", f), false);
});
test("eyelids: a wink does not count", () =>
  assert.equal(run("blink", [lids(1), lids(0.1, 1), lids(1), lids(1, 0.1), lids(1)]), false));
test("eyelids: eyes that stay shut do not pass", () =>
  assert.equal(run("blink", [lids(1), lids(0.1), lids(0.1), lids(0.15)]), false));
test("faceMetrics: eyelid openness is height / width", () => {
  const m = G.faceMetrics(faceRes({ openL: 1, openR: 0.5 }));
  assert.ok(Math.abs(m.earL - 0.3) < 1e-6 && Math.abs(m.earR - 0.15) < 1e-6);
});

// ---- the capture loop's gate: the challenge only steps when there is no problem
const loop = (type, frames) => {
  const c = G.createChallenge(type);
  for (const f of frames) {
    const face = G.faceMetrics(faceRes(f));
    if (!G.liveFaceProblem(face, { challenge: c })) c.step(face);
  }
  return c;
};
test("loop: the turn challenge can actually be completed", () =>
  assert.equal(loop("turn", [{ yaw: 0 }, { yaw: 0.25 }, { yaw: 0.02 }]).done, true));
test("loop: the blink challenge can actually be completed", () =>
  assert.equal(loop("blink", [open, shut, open]).done, true));
test("loop: once done, a turned head is asked to face front", () => {
  const c = loop("turn", [{ yaw: 0 }, { yaw: 0.25 }, { yaw: 0.02 }]);
  assert.equal(G.liveFaceProblem(G.faceMetrics(faceRes({ yaw: 0.25 })), { challenge: c }), "Look straight at the camera");
});
test("loop: two faces block, and the challenge never steps", () =>
  assert.equal(loop("blink", [{ ...open, faces: 2 }, { ...shut, faces: 2 }, { ...open, faces: 2 }]).done, false));

console.log(`guide.js: ${passed} tests passed`);
