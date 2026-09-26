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
  LsnScoreOptions,
  LsnScores,
  LsnStartOptions,
  LsnStartResult,
  LsnWarmUpResult,
} from './types';

export * from './types';
export { toScores } from './scores';

function native(): NativeLsnCaptureSpec {
  if (!NativeLsnCapture) {
    throw lsnError(
      'unsupported_platform',
      'LSN Capture is Android-only (or the native module is not linked)',
    );
  }
  return NativeLsnCapture;
}

/** true on Android when the native module is linked. */
export function isSupported(): boolean {
  return NativeLsnCapture != null;
}

/** Load the face model and CameraX ahead of time (call when the screen before the capture shows). */
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
    const r = await m.capture({
      challenge: opts.challenge ?? 'blink',
      light: opts.light ?? false,
      maxSide: opts.maxSide ?? 2592,
      jpegQuality: opts.jpegQuality ?? 92,
      brightness: opts.brightness ?? 0,
    });
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
  const {
    challenge,
    light,
    maxSide,
    jpegQuality,
    brightness,
    ...scoreOpts
  } = opts;
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
