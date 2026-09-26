import { NativeLsnCapture } from './NativeLsnCapture';
import type { NativeLsnCaptureSpec } from './NativeLsnCapture';
import { lsnError, toLsnError } from './errors';
import {
  ALL_CHECKS,
  DEFAULT_TIMEOUT_MS,
  toScores,
  validateScoreOptions,
} from './scores';
import type {
  LsnCaptureOptions,
  LsnCaptureResult,
  LsnChallenge,
  LsnScoreOptions,
  LsnScores,
  LsnStartOptions,
  LsnStartResult,
  LsnWarmUpResult,
} from './types';

export * from './types';
export { toScores } from './scores';

const CHALLENGES: readonly LsnChallenge[] = ['none', 'blink', 'turn', 'random'];

function native(): NativeLsnCaptureSpec {
  if (!NativeLsnCapture) {
    throw lsnError(
      'unsupported_platform',
      'LSN Capture is Android-only (or the native module is not linked)',
    );
  }
  return NativeLsnCapture;
}

const finite = (v: unknown): number | undefined =>
  typeof v === 'number' && Number.isFinite(v) ? v : undefined;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** The capture options as sent to native: defaults filled in, bad values coerced (never throws). */
function captureArgs(opts: LsnCaptureOptions | null | undefined) {
  const o = opts ?? {};
  const c = typeof o.challenge === 'string' ? o.challenge.trim().toLowerCase() : '';
  return {
    challenge: (CHALLENGES as readonly string[]).includes(c) ? c : 'blink',
    light: o.light === true,
    maxSide: Math.max(0, Math.round(finite(o.maxSide) ?? 2592)),
    jpegQuality: clamp(Math.round(finite(o.jpegQuality) ?? 92), 60, 100),
    brightness: clamp(finite(o.brightness) ?? 0, -2, 2),
  };
}

/** true on Android when the native module is linked. Never throws. */
export function isSupported(): boolean {
  return NativeLsnCapture != null;
}

/**
 * Load the face model and CameraX ahead of time (call when the screen before
 * the capture shows). Resolves with modelReady=false instead of rejecting when
 * the model is not there; can take long on first run (model download), so
 * don't await it on the critical path.
 */
export async function warmUp(): Promise<LsnWarmUpResult> {
  const m = native();
  try {
    return await m.warmUp();
  } catch (e) {
    throw toLsnError(e, 'capture_failed');
  }
}

/** Open the native camera screen; resolves with the JPEG path. Rejects 'cancelled' if the rider backs out. */
export async function capture(
  opts: LsnCaptureOptions = {},
): Promise<LsnCaptureResult> {
  const m = native();
  try {
    const r = await m.capture(captureArgs(opts));
    return {
      path: r.path,
      uri: r.uri ?? `file://${r.path}`,
      challenge: r.challenge,
      challengePassed: !!r.challengePassed,
      stats: r.stats ?? {},
    };
  } catch (e) {
    throw toLsnError(e, 'capture_failed');
  }
}

/** POST {apiBase}/v1/checkpoint/score. The file is read and base64-encoded natively. */
export async function score(opts: LsnScoreOptions): Promise<LsnScores> {
  const m = native();
  validateScoreOptions(opts, true);
  try {
    const raw = await m.score({
      apiBase: opts.apiBase,
      checks: opts.checks ?? ALL_CHECKS,
      faceCheckPath: opts.faceCheckPath,
      faceCheckS3: opts.faceCheckS3,
      sourceSelfieS3: opts.sourceSelfieS3,
      sourceSelfieB64: opts.sourceSelfieB64,
      timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      headers: opts.headers ?? {},
    });
    return toScores(raw);
  } catch (e) {
    throw toLsnError(e, 'network');
  }
}

/** capture() then score() on the captured photo. Score options are validated before the camera opens. */
export async function start(opts: LsnStartOptions): Promise<LsnStartResult> {
  native();
  validateScoreOptions(opts);
  const { challenge, light, maxSide, jpegQuality, brightness, ...rest } = opts;
  // The captured photo is the one scored: drop any photo source the caller passed.
  const { faceCheckPath: _p, faceCheckS3: _s, ...scoreOpts } =
    rest as LsnScoreOptions;
  const shot = await capture({
    challenge,
    light,
    maxSide,
    jpegQuality,
    brightness,
  });
  const scores = await score({ ...scoreOpts, faceCheckPath: shot.path });
  return { capture: shot, scores };
}
