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
 * SDK's LsnScores). Pure: never throws, missing fields become null.
 */
export function toScores(raw: Record<string, unknown>): LsnScores {
  const json: Obj = isObj(raw) ? raw : {};
  const scores = isObj(json.scores) ? json.scores : {};
  const s = (k: string): Obj => (isObj(scores[k]) ? (scores[k] as Obj) : {});
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

/**
 * Same rules as the native side (and the Flutter SDK's api.dart), checked in JS
 * too so `start()` fails before the rider is sent to the camera.
 * Throws an LsnError with code 'invalid_config'.
 */
export function validateScoreOptions(
  opts: Omit<LsnScoreOptions, 'faceCheckPath' | 'faceCheckS3'>,
): void {
  const base = (opts?.apiBase ?? '').trim().replace(/\/+$/, '');
  if (!/^https?:\/\/[^/?#\s]+/i.test(base)) {
    throw lsnError('invalid_config', 'apiBase must be an http(s) URL');
  }
  const checks = opts.checks ?? ALL_CHECKS;
  if (checks.length === 0 || checks.some((c) => !ALL_CHECKS.includes(c))) {
    throw lsnError(
      'invalid_config',
      `checks must be a non-empty subset of ${ALL_CHECKS.join(', ')}`,
    );
  }
  if (
    checks.includes('face_match') &&
    !opts.sourceSelfieS3 &&
    !opts.sourceSelfieB64
  ) {
    throw lsnError(
      'invalid_config',
      'face_match needs sourceSelfieS3 or sourceSelfieB64',
    );
  }
  if (opts.timeoutMs !== undefined && !(opts.timeoutMs > 0)) {
    throw lsnError('invalid_config', 'timeoutMs must be > 0');
  }
}
