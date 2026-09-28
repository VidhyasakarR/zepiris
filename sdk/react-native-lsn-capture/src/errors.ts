import type { LsnError, LsnErrorCode } from './types';

export const LSN_ERROR_CODES: readonly LsnErrorCode[] = [
  'cancelled',
  'busy',
  'no_activity',
  'no_permission',
  'face_check_unavailable',
  'capture_failed',
  'invalid_config',
  'network',
  'timeout',
  'server',
  'unsupported_platform',
];

const isLsnCode = (c: unknown): c is LsnErrorCode =>
  typeof c === 'string' && (LSN_ERROR_CODES as readonly string[]).includes(c);

export function lsnError(code: LsnErrorCode, message?: string): LsnError {
  const e = new Error(message || code) as LsnError;
  e.name = 'LsnError';
  e.code = code;
  return e;
}

/**
 * A native promise rejection (or anything thrown) → an Error whose `.code` is
 * an LsnErrorCode. An unknown code becomes `fallback`; the original code is
 * kept in `.nativeCode` and the original message is never replaced.
 */
export function toLsnError(e: unknown, fallback: LsnErrorCode): LsnError {
  const src = (typeof e === 'object' && e !== null ? e : {}) as {
    code?: unknown;
    message?: unknown;
  };
  const code = isLsnCode(src.code) ? src.code : fallback;
  if (e instanceof Error) {
    const err = e as LsnError;
    if (!isLsnCode(src.code) && src.code != null) {
      err.nativeCode = String(src.code);
    }
    err.code = code;
    if (!err.message) err.message = code;
    return err;
  }
  const message =
    typeof src.message === 'string' && src.message
      ? src.message
      : typeof e === 'string' && e
        ? e
        : code;
  const err = lsnError(code, message);
  if (!isLsnCode(src.code) && src.code != null) {
    err.nativeCode = String(src.code);
  }
  return err;
}
