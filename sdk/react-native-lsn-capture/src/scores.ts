import type { LsnCheck, LsnScoreOptions, LsnScores } from './types';
import { lsnError } from './errors';

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const str = (v: unknown): string | null => (typeof v === 'string' ? v : null);

/**
 * The raw /v1/checkpoint/score JSON, flattened (same mapping as the Flutter
 * SDK's LsnScores). Pure: never throws; missing or wrongly-typed fields
 * (strings for numbers, NaN, arrays for objects) become null.
 */
export function toScores(raw: unknown): LsnScores {
  const json: Obj = isObj(raw) ? raw : {};
  const scores = isObj(json.scores) ? json.scores : {};
  const s = (k: string): Obj => {
    const v = scores[k];
    return isObj(v) ? v : {};
  };
  const fm = s('face_match');
  const logo = s('logo');
  const image = isObj(json.image) ? json.image : {};
  return {
    requestId: str(json.requestId) ?? '',
    scoredAt: str(json.scoredAt),
    checksRequested: Array.isArray(json.checksRequested)
      ? json.checksRequested.filter((c): c is string => typeof c === 'string')
      : [],
    faceSimilarity: num(fm.similarity),
    liveness: num(fm.liveness),
    faceDetected: typeof fm.faceDetected === 'boolean' ? fm.faceDetected : null,
    dressColor: num(s('dress_color').score),
    logo: num(logo.score),
    logoRead: isObj(logo.read) ? logo.read : null,
    imageSha256: str(image.sha256),
    raw: json,
  };
}

export const ALL_CHECKS: LsnCheck[] = ['face_match', 'dress_color', 'logo'];
export const DEFAULT_TIMEOUT_MS = 40000;
/** Keep in sync with ScoreRequest in android/.../ScoreClient.kt. */
export const MAX_TIMEOUT_MS = 600000;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
export const MAX_SELFIE_B64_CHARS = Math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 64;
const HEADER_NAME = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;
// eslint-disable-next-line no-control-regex
const HEADER_VALUE = /^[\t\x20-\x7e]*$/;

const bad = (msg: string): never => {
  throw lsnError('invalid_config', msg);
};
const nonEmpty = (v: unknown): boolean => typeof v === 'string' && v !== '';

/**
 * Same rules as the native side (ScoreRequest.validate, itself the Flutter
 * SDK's api.dart), checked in JS too so errors surface before the bridge and
 * `start()` fails before the rider is sent to the camera.
 * `requirePhoto: false` skips the faceCheckPath / faceCheckS3 rule (start()).
 * Throws an LsnError with code 'invalid_config'.
 */
export function validateScoreOptions(
  opts: Partial<LsnScoreOptions> | null | undefined,
  requirePhoto = false,
): void {
  if (!isObj(opts)) bad('score options are required');
  const o = opts as Partial<LsnScoreOptions>;
  const base = (typeof o.apiBase === 'string' ? o.apiBase : '')
    .trim()
    .replace(/\/+$/, '');
  if (!/^https?:\/\/[^/?#\s]+(\/[^?#\s]*)?$/i.test(base)) {
    bad('apiBase must be an http(s) URL without a query or fragment');
  }
  const checks: unknown = o.checks ?? ALL_CHECKS;
  if (
    !Array.isArray(checks) ||
    checks.length === 0 ||
    checks.some((c) => !ALL_CHECKS.includes(c as LsnCheck))
  ) {
    bad(`checks must be a non-empty subset of ${ALL_CHECKS.join(', ')}`);
  }
  if (
    (checks as LsnCheck[]).includes('face_match') &&
    !nonEmpty(o.sourceSelfieS3) &&
    !nonEmpty(o.sourceSelfieB64)
  ) {
    bad('face_match needs sourceSelfieS3 or sourceSelfieB64');
  }
  if (
    typeof o.sourceSelfieB64 === 'string' &&
    o.sourceSelfieB64.length > MAX_SELFIE_B64_CHARS
  ) {
    bad("sourceSelfieB64 is larger than the server's 5 MB image limit");
  }
  if (requirePhoto && nonEmpty(o.faceCheckPath) === nonEmpty(o.faceCheckS3)) {
    bad('pass exactly one of faceCheckPath or faceCheckS3');
  }
  if (
    o.timeoutMs !== undefined &&
    !(
      typeof o.timeoutMs === 'number' &&
      o.timeoutMs >= 1 &&
      o.timeoutMs <= MAX_TIMEOUT_MS
    )
  ) {
    bad(`timeoutMs must be > 0 and <= ${MAX_TIMEOUT_MS}`);
  }
  if (o.headers !== undefined) {
    if (!isObj(o.headers)) bad('headers must be an object of strings');
    for (const [name, value] of Object.entries(o.headers as Obj)) {
      if (!HEADER_NAME.test(name)) bad('headers: invalid header name');
      if (typeof value !== 'string') bad(`headers: the value of ${name} must be a string`);
      if (!HEADER_VALUE.test(value as string)) {
        bad(
          `headers: the value of ${name} has a control (CR/LF) or non-ASCII character`,
        );
      }
    }
  }
}
