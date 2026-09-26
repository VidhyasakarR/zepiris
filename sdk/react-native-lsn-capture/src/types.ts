/** Liveness challenge the rider completes before the photo (random = blink or turn). */
export type LsnChallenge = 'none' | 'blink' | 'turn' | 'random';

/** Server-side checks of POST /v1/checkpoint/score. */
export type LsnCheck = 'face_match' | 'dress_color' | 'logo';

export type LsnErrorCode =
  | 'cancelled' // the rider backed out of the camera screen
  | 'busy' // a capture is already running
  | 'no_activity' // no foreground Android activity
  | 'no_permission' // CAMERA permission not granted (the host must request it)
  | 'face_check_unavailable' // the Play Services face model never became available
  | 'capture_failed' // the camera screen could not open / the camera failed
  | 'invalid_config' // bad score options (apiBase, checks, photo source, selfie)
  | 'network' // the server could not be reached
  | 'timeout' // the server took longer than timeoutMs
  | 'server' // non-2xx, or a reply without scores
  | 'unsupported_platform'; // not Android, or the native module is not linked

export interface LsnError extends Error {
  code: LsnErrorCode;
}

export interface LsnCaptureOptions {
  /** Default 'blink'. */
  challenge?: LsnChallenge;
  /** Start with the screen light on (default false). */
  light?: boolean;
  /** Long side of the photo in px; 0 = camera's full resolution (default 2592). */
  maxSide?: number;
  /** JPEG quality 60-100 (default 92). */
  jpegQuality?: number;
  /** Starting exposure compensation in EV, -2..+2 (default 0). */
  brightness?: number;
}

export interface LsnCaptureResult {
  /** Absolute path of the JPEG in the app cache (not deleted by the SDK). */
  path: string;
  /** 'file://' + path */
  uri: string;
  /** The challenge that ran (random resolved to blink / turn). */
  challenge: string;
  challengePassed: boolean;
  /** On-device stats: firstDetectionMs, captureMs, frames, width, height, bytes, ... */
  stats: Record<string, unknown>;
}

export interface LsnScoreOptions {
  /** Server root, e.g. https://3-108-193-187.sslip.io (trailing slashes dropped). */
  apiBase: string;
  /** Default all three. */
  checks?: LsnCheck[];
  /** Local JPEG to score (plain path or file:// URI). Exactly one of faceCheckPath / faceCheckS3. */
  faceCheckPath?: string;
  /** S3/HTTP link of the photo to score. */
  faceCheckS3?: string;
  /** Enrolled selfie for face_match (S3/HTTP link). face_match needs this or sourceSelfieB64. */
  sourceSelfieS3?: string;
  /** Enrolled selfie for face_match (base64 JPEG). */
  sourceSelfieB64?: string;
  /** Default 40000. */
  timeoutMs?: number;
  /** Extra request headers (e.g. a gateway token). */
  headers?: Record<string, string>;
}

/** /v1/checkpoint/score, flattened. Scores only: pass / fail is the host backend's call. */
export interface LsnScores {
  requestId: string;
  scoredAt: string | null;
  checksRequested: string[];
  /** Cosine similarity to the enrolled selfie (null: no face / not requested). */
  faceSimilarity: number | null;
  /** Probability of a live person (null when server liveness is off). */
  liveness: number | null;
  faceDetected: boolean | null;
  dressColor: number | null;
  logo: number | null;
  /** What the logo check read: {text, matched, onBlue, reason}. */
  logoRead: Record<string, unknown> | null;
  /** SHA-256 (hex) of the exact photo that was scored. */
  imageSha256: string | null;
  /** The response exactly as the server sent it. */
  raw: Record<string, unknown>;
}

export interface LsnWarmUpResult {
  modelReady: boolean;
  ms: number;
  reason?: string | null;
}

export type LsnStartOptions = LsnCaptureOptions &
  Omit<LsnScoreOptions, 'faceCheckPath' | 'faceCheckS3'>;

export interface LsnStartResult {
  capture: LsnCaptureResult;
  scores: LsnScores;
}
