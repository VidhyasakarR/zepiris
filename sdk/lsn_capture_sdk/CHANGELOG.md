## 1.6.3

- The ☀ brightness slider is back on the camera screen (camera exposure compensation, adjustable while the camera runs; hidden on cameras without it). `brightness` sets where it starts.

## 1.6.2

- Screen light: while it is on, the 💡 button reads "💡 Turn off ✕", so riders see how to switch it off.

## 1.6.1

- Dim light no longer blocks: if the face (and T-shirt) can be detected and framed, Capture unlocks, with "Low light — tap 💡" (or "photo may be dark" with the light on) as a warning. Only a nearly black face (luma < 15) blocks. The server's quality check still flags a too-dark photo after capture.
- The ☀ brightness slider is removed from the camera screen (it did not help in the dark). `brightness` still sets the starting exposure.
- Screen light: the bars around the camera picture turn white too (they stayed black and cut the light).

## 1.6.0

- **Breaking:** the automatic low-light screen flash is gone, with `flash` and `flashDuration`. In its place:
- Screen light 💡 on the camera screen: tap it and the camera picture shrinks while the white around it, at full screen brightness, lights the face (a ring light). The camera adapts its exposure on its own, and the rider captures as usual. Tap again to turn it off. `LsnCaptureConfig(screenLight: true)` starts with it on. In the dark the hint says "Low light — tap 💡".
- Brightness slider (☀) in place of − / +: drag to brighten or darken the camera (exposure compensation) while it runs; `brightness` sets where it starts.
- `stats`: `flashed` is replaced by `light` (was the screen light on).

## 1.5.0

- Light is judged on the face, not the whole frame (as the server's quality check does): a well-lit face in a dark room is no longer blocked as "Too dark". With no face in view, the whole frame is used.
- The T-shirt needs light too (the dress-colour check reads it): "T-shirt too dark — turn toward the light" when the chin-to-stomach region's brighter parts (75th percentile) are below 35. A percentile, so a dark shirt in good light is not blocked; judging the colour is the colour check's job.
- Camera brightness: `LsnCaptureConfig(brightness: 1)` starts the camera's exposure compensation at +1 EV (-2 .. +2), and − / + buttons on the camera screen let the rider adjust it. It brightens the preview and so the photo itself. Hidden on cameras without exposure compensation. `stats` adds `evIndex`.

## 1.4.1

- The photo crop and the oval guide follow the camera's actual preview shape (from its resolution), not an assumed 3:4, so a camera that falls back to another aspect ratio gets no black bars in the photo and no offset guide.
- Orientation self-check decides only on a clear difference (a plain wall or a dark room look the same either way round) and otherwise retries, up to 20 times, on later frames.
- `stats` adds `cameraOpenMs` (screen opened to camera bound) and `encodeMs` (JPEG encode after the tap, runs behind the frozen frame).

## 1.4.0

- Zero-delay capture: the photo is the preview frame on screen at the instant Capture is tapped (PreviewView.getBitmap). No second camera capture, no shutter, no "Capturing…" wait. The frame freezes on screen immediately and is encoded behind it. A full ImageCapture shot is only the fallback if the preview cannot be read.
- Self-calibrating orientation: once per session the SDK compares a preview frame with an analysis frame (whose orientation is known), straight vs flipped, so the photo is the true scene on every phone model. `stats` adds `source`, `previewMirrored`, `tapToShotMs`.
- Photo size is the preview's (1080×1440 on a 1080-px-wide phone). `photoQuality` sets JPEG quality and caps the size.

## 1.3.1

- Instant capture: the shutter fires the moment Capture is tapped. The "wait for a still moment" step is gone: riders move right after tapping, so waiting made photos blurrier and framed differently from what was tapped.
- Zero shutter lag where the camera supports it (CameraX `CAPTURE_MODE_ZERO_SHUTTER_LAG`): the photo is the frame from the instant of the tap, from the camera's ring buffer, before the tap shakes the phone. Otherwise the fastest normal capture (`MINIMIZE_LATENCY`), not the multi-frame quality mode.
- With the optional flash, only its on-time (`flashDuration`) is waited.
- `stats` adds `zsl` (was zero shutter lag used) and `tapToShotMs`.

## 1.3.0

- Photos no longer come out upside down: each shot is straightened with the camera's own rotation for that frame (the value the live face detection uses), not the JPEG's EXIF tag, which is 180° off on some front cameras. Never mirrored. `stats` now reports `rotation`, `exifOrientation`, `width`, `height`, `jpegQuality`, `bytes`.
- `photoQuality`: `LsnPhotoQuality.standard` (1920 px, JPEG 90), `.high` (2592 px, JPEG 92, the default) or `.max` (the camera's full resolution, JPEG 95). JPEG quality steps down only if a photo would pass 4.5 MB (the server's limit is 5 MB).
- Sharper photos: CameraX quality capture mode (full ISP processing) instead of the fast path, one JPEG encode at the chosen quality, and the review image is drawn with high-quality filtering.
- Clear waits: "Capturing…" then "Processing photo…" on the camera screen, and a spinner with "Checking photo quality…" / "Getting your scores…" on the review screen (no more "Hold still…" left on screen).

## 1.2.0

- Sharper photos: after Capture is tapped the shutter waits for a still moment (face steady for ~0.3 s, 2.25 s at most), so the tap's shake and a moving hand no longer blur the shot.
- Shorter exposures: the camera's auto-exposure is asked to keep >= ~24 fps (shutter ~1/24 s or faster), trading a little grain for less motion blur. Skipped on cameras that offer no such range.
- Low-light screen flash is now optional and off by default: `LsnCaptureConfig(flash: true, flashDuration: Duration(milliseconds: 800))`. When on, it uses CameraX's screen-flash mode (exposure metered with the white screen lit), shows "Hold still…", takes two shots and keeps the sharper.

## 1.1.0

- Scores, not decisions: the SDK calls `/v1/checkpoint/score` and returns `LsnScores`; pass / fail is the host backend's call.
- Typed `LsnChallenge` (blink / turn / random / none).
- `headers` (e.g. an auth token), `httpClient` (e.g. a network inspector), `qualityTimeout` and `scoreTimeout` on `LsnCaptureConfig`; the config is validated when built.

## 1.0.1

- Photos are upright: the camera's EXIF rotation is baked into the pixels right after capture, so the review screen and the server see the same image without relying on EXIF handling.
- No mirroring anywhere: the live preview shows the true scene (Android mirrors the front camera by default), matching the photo that is sent, so T-shirt text reads normally.

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
