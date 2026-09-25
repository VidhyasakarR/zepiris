## 1.1.0

- **Scores only.** The SDK calls `/v1/checkpoint/score` and returns `LsnCaptureResult.scores` (`LsnScores`: typed getters plus the raw `json`). There is no pass / fail anywhere; the host's backend decides. `isCleared` and the result screen are removed.
- **Cleaner flow:** capture → quality check → scores → back to the host. It stops only for a quality warning (Retake / Use anyway) or a failed call (Retake / Try again).
- **Config:** `LsnChallenge` enum, validation up front (`ArgumentError`), `headers`, an injectable `httpClient` (for interceptors and inspectors such as Chucker; never closed by the SDK), and `qualityTimeout` / `scoreTimeout`.
- **Android fixes:**
  - No crash when the screen closes mid-detection.
  - Edge-to-edge insets keep Capture above the navigation bar on Android 15+.
  - Predictive back is handled.
  - Old temp selfies are cleaned up, and the photo file is always deleted after reading.
  - The per-frame failure log only appears in debug.
  - A failed launch no longer leaves the plugin "busy".
  - A blink that never reopens (about 1.5 s) resets.
  - The prompt reads "Blink 3 times, quickly".

## 1.0.0

- Native Android capture screen: CameraX preview and 4:3 photo (1440×1920), with 640×480 analysis frames going straight to ML Kit face detection from Play Services.
- Live checks: light, one face, size, centring, T-shirt in frame (the server's capture-quality rule), head frontal, eyes open, holding still.
- Liveness challenge: blink (judged against the rider's own open-eye level) or head turn, restarted when the face changes.
- Low-light screen flash at full brightness.
- Review screen: server quality warnings, Retake or Submit, the checkpoint result.
- `LsnCapture.warmUp()` / `LsnCapture.start()` API; `submit: false` returns the photo only.
