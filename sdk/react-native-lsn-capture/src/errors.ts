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

export function lsnError(code: LsnErrorCode, message?: string): LsnError {
  const e = new Error(message || code) as LsnError;
  e.name = 'LsnError';
  e.code = code;
  return e;
}

/** A native promise rejection (or anything thrown) → an Error whose `.code` is an LsnErrorCode. */
export function toLsnError(e: unknown, fallback: LsnErrorCode): LsnError {
  const anyE = e as { code?: unknown; message?: unknown } | null | undefined;
  const code =
    typeof anyE?.code === 'string' &&
    (LSN_ERROR_CODES as readonly string[]).includes(anyE.code)
      ? (anyE.code as LsnErrorCode)
      : fallback;
  const message =
    typeof anyE?.message === 'string' && anyE.message ? anyE.message : code;
  if (e instanceof Error) {
    const err = e as LsnError;
    err.code = code;
    return err;
  }
  return lsnError(code, message);
}
