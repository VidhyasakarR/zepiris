# Changelog

## 1.0.0 — 2026-09-26

First release: the React Native port of the Flutter `lsn_capture_sdk` (1.6.3) capture screen.

- `capture()`: native Android camera screen (`CaptureActivity`, copied from the Flutter plugin) with the oval guide, live face / T-shirt checks and blink / turn liveness. Resolves with the JPEG path; the file is left in the app cache for the host to upload.
- `score()`: `POST {apiBase}/v1/checkpoint/score` from Kotlin (OkHttp), so the photo never crosses the JS bridge. Same validation and error parsing as the Flutter SDK's `api.dart`.
- `start()`: `capture()` then `score()`.
- `warmUp()`: loads the Play Services face model and CameraX ahead of time.
- `toScores()`: flattens the raw score JSON (`faceSimilarity`, `liveness`, `dressColor`, `logo`, ...).
- Errors are `Error` objects with `.code` (`cancelled`, `busy`, `no_permission`, `timeout`, `server`, ...). Non-Android platforms reject with `unsupported_platform`.
