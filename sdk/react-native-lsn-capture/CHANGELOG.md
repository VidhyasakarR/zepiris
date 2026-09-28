# Changelog

## 1.0.2 — 2026-09-26

Hardening from a production-readiness review. The public API is unchanged; additions are optional (`LsnError.nativeCode`, `toScores` accepting `unknown`).

- **Capture lifecycle:** a capture can no longer leave the module stuck on `busy`. If the camera screen closes and its result never reaches the module, the capture rejects with `capture_failed`. The screen is now launched on the main thread. A React context teardown drops the pending capture and unregisters the listeners. Process death / activity recreation behaviour is documented.
- **Capture options:** `challenge` is case-insensitive and an unknown value falls back to `'blink'` (it used to silently disable liveness). `NaN` / out-of-range `maxSide`, `jpegQuality` and `brightness` are replaced or clamped, in JS and natively.
- **`score()` validation (JS and Kotlin, same rules):** `apiBase` must have no query / fragment; `timeoutMs` 1..600000; `headers` must be RFC 9110 names with printable-ASCII values (CR/LF injection rejected, values never echoed in errors), and `Content-Type` / `Content-Length` / `Host` / `Connection` / `Transfer-Encoding` are ignored; `sourceSelfieB64` and `faceCheckPath` capped at the server's 5 MiB (missing / empty / too-large files reject with `invalid_config` before any network call); duplicate checks are collapsed. `score()` now validates in JS before calling native.
- **`score()` request:** the photo's base64 is spliced into the JSON body as bytes (about half the peak memory of the old string path); connect timeout capped at 15 s (`timeoutMs` still bounds the whole call); `https` → `http` redirects are not followed; the reply is capped at 1 MB; cleartext blocked by the app's network security config rejects with `invalid_config` instead of a generic `network`; the `server` message is capped at 300 chars; `User-Agent` is `LSNCaptureSDK-RN/<package.json version>` (generated `BuildConfig.SDK_VERSION`). The score promise always settles, even on `OutOfMemoryError`.
- **`start()`:** ignores any `faceCheckPath` / `faceCheckS3` passed in (it used to send both and fail with `invalid_config` after the photo was taken).
- **Errors:** an unknown native code maps to the function's fallback with the original kept in `.nativeCode`; the message is never replaced. `isSupported()` never throws (no native runtime, e.g. Jest / Expo Go).
- **Build:** consumer R8 rules narrowed to the module, package and activity. `sideEffects: false`.
- **Tests:** Jest 5 → 56 (errors, validation, JS API with a mocked native module, defensive `toScores`); `ScoreClientTest` 19 → 58 (header sanitising, limits, body building, reply cap, MockWebServer end-to-end on OkHttp 4.9.2).
- **Docs:** README covers threading, lifecycle, privacy, file ownership, size limits, HTTP policy and the `LsnBaseActivity` shim.

## 1.0.1 — 2026-09-26

- Fix: Kotlin compile error `CaptureActivity is not abstract and does not implement abstract member getViewModelStore` in host apps that force `androidx.lifecycle:lifecycle-viewmodel` below 2.6 (titan-rider-app forces 2.5.1) while `androidx.activity` is 1.9.x. `CaptureActivity` now extends a small Java shim, `LsnBaseActivity`, which re-declares the ViewModel owner getters. No behaviour change.

## 1.0.0 — 2026-09-26

First release: the React Native port of the Flutter `lsn_capture_sdk` (1.6.3) capture screen.

- `capture()`: native Android camera screen (`CaptureActivity`, copied from the Flutter plugin) with the oval guide, live face / T-shirt checks and blink / turn liveness. Resolves with the JPEG path; the file is left in the app cache for the host to upload.
- `score()`: `POST {apiBase}/v1/checkpoint/score` from Kotlin (OkHttp), so the photo never crosses the JS bridge. Same validation and error parsing as the Flutter SDK's `api.dart`.
- `start()`: `capture()` then `score()`.
- `warmUp()`: loads the Play Services face model and CameraX ahead of time.
- `toScores()`: flattens the raw score JSON (`faceSimilarity`, `liveness`, `dressColor`, `logo`, ...).
- Errors are `Error` objects with `.code` (`cancelled`, `busy`, `no_permission`, `timeout`, `server`, ...). Non-Android platforms reject with `unsupported_platform`.
