# LSN Capture SDK

The Loadshare checkpoint selfie as a native Android flow in a Flutter package. It runs the camera, the oval guide, live checks and a liveness challenge, and captures a photo. It then runs the server's photo quality check and submits to `/v1/checkpoint/score`. It shows no pass / fail: it returns the scores, and the host's backend decides.

## Use it

```yaml
dependencies:
  lsn_capture_sdk:
    path: ../lsn_capture_sdk   # wherever you put the folder (see ../README.md for git / zip)
```

```dart
import 'package:lsn_capture_sdk/lsn_capture_sdk.dart';

// 1. Early, for example when the screen before the capture shows:
LsnCapture.warmUp(); // face model ready + CameraX initialised

// 2. The app must hold the CAMERA permission, then:
final r = await LsnCapture.start(context, LsnCaptureConfig(
  apiBase: 'https://3-108-193-187.sslip.io',
  checks: ['face_match', 'dress_color', 'logo'],
  sourceSelfieS3: 'https://bucket.s3.amazonaws.com/rider.jpg', // or sourceSelfieB64
  challenge: LsnChallenge.blink,                               // blink | turn | random | none
  headers: {'Authorization': 'Bearer …'},                       // optional, for your gateway
  httpClient: myClient,                                         // optional: your interceptors / Chucker
));
if (r != null) {                 // null: the rider backed out
  final s = r.scores!;           // LsnScores: scores only, no pass / fail
  s.faceSimilarity; s.liveness; s.dressColor; s.logo; s.logoRead;
  s.json;                        // the server's response, unmodified: send this to your backend
  r.photoJpeg;                   // the exact photo scored (SHA-256 = s.imageSha256): upload it
  r.challenge; r.challengePassed; r.quality; r.stats;
}
```

**The flow:** camera → capture → photo quality check → scores → return. It stops on the photo only when the rider has a choice to make:
- a quality warning (blur, light, T-shirt not in view): **Retake** or **Use anyway**;
- a failed scoring call: **Retake** or **Try again**.

A clear photo goes straight to scoring and back to the host.

**Config:**
- **Validated up front.** `LsnCaptureConfig` throws `ArgumentError` for a bad `apiBase`, unknown `checks`, or `face_match` without a source selfie.
- **`submit: false`** returns the photo unscored, so the host can score it itself.
- **`qualityCheck: false`** skips `/v1/quality/check`.
- **`qualityTimeout` / `scoreTimeout`** default to 8 s and 40 s.
- **`httpClient`** is never closed by the SDK. It closes only the client it created itself.

**`r.scores.json`** is `/v1/checkpoint/score`: `{requestId, scoredAt, checksRequested, scores, image: {sha256, bytes}}`. The field reference is in the repo's `LSN_README.md` ("Scores only").

**Errors:**
- Camera problems end the flow with a message: `no_permission`, `face_check_unavailable` (no Play Services), or camera unavailable.
- Server errors are shown with Try again. On the Dart side they are `LsnApiError` (`message`, `status`).

## How it works

| Part | What it does |
|---|---|
| `CaptureActivity.kt` | CameraX with three use cases, each at its own resolution. **Preview** and **photo** are 4:3, the camera's native shape and the framing of the phone's own camera app. The photo is 1920×1440 (portrait 1440×1920). **Analysis** is 640×480, in the same 4:3 field of view. The preview is fitted, not cropped, so the rider sees exactly what the photo contains. |
| `FaceEngine.kt` | ML Kit face detection in fast mode, with eye-open classification and tracking. It comes from Google Play Services, so the APK only grows by about 0.8 MB. `warmUp` asks Play Services to install the model if it's missing, runs one dummy frame, and initialises CameraX. |
| `Checks.kt` | The live checks, in order: light, exactly one face, size, centred, enough T-shirt in frame, head frontal, eyes open, holding still. Also the liveness challenge. The T-shirt rule is the server's own (`capture_quality.py`): at least 35% of the region from the chin down to 3 face heights must be in frame. Covered by JUnit tests in `src/test`. |
| `capture_screen.dart` | After the photo: quality warnings, scoring, then back to the host (Retake / Use anyway / Try again when needed). |

Liveness:
- **Blink:** measured against the rider's own open-eye level, so glasses and dim light work. A wink or a still photo never passes.
- **Turn:** turn more than 20° to one side, then come back within 8°.
- **Face swaps:** a second face, or 1.2 s with no face, restarts the challenge.
- **Low light:** the screen turns white for 0.8 s while the photo is taken, at full brightness.

## Numbers

Measured on a Redmi 2312FRAFDI (Android 15), arm64 release APK:

| Build | APK |
|---|---|
| App without the SDK | 17.95 MB |
| With this SDK (CameraX + ML Kit via Play Services) | **18.78 MB (+0.82 MB; +0.90 MB on armeabi-v7a)** |
| The SDK's own compiled code (`.aar`) | 42 KB |
| Earlier Dart version with the bundled ML Kit model | 32.0 MB |

## Tests

```bash
cd <your app>/android && ./gradlew :lsn_capture_sdk:testReleaseUnitTest   # e.g. mobile/lsn_checkpoint
```

Face measurements go to logcat (tag `LsnCapture`) when the host builds with `--dart-define=LSN_SDK_DEBUG=true`.

## Limits

- **Android only.** An iOS version would be the same design in Swift, with AVFoundation and ML Kit for iOS.
- **Needs Google Play Services** for the face model. Without it, the flow ends with `face_check_unavailable` after 10 s; use the web flow there.
- **Liveness is checked on the phone only.** `challengePassed` is returned to the app, but the server doesn't receive it. The server's own passive liveness check still decides.
