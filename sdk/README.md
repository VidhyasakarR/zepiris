# Loadshare SDKs

## LSN Capture SDK (`lsn_capture_sdk/`, v1.6.3)

The checkpoint selfie as a drop-in Flutter plugin with a native Android camera screen. It covers the oval guide, live face and T-shirt checks, and blink or head-turn liveness. It then runs the server's photo quality check and `/v1/checkpoint/verify`, and shows the result. Full API and design: [`lsn_capture_sdk/README.md`](lsn_capture_sdk/README.md).

### Size

| | |
|---|---|
| Added to an app (arm64 APK) | **+0.82 MB**: 17.95 MB → 18.78 MB, measured |
| Added to an app (armeabi-v7a APK) | +0.90 MB |
| SDK's own compiled code (`.aar`) | 42 KB |
| Source package (`dist/lsn_capture_sdk-1.6.3.zip`) | 31 KB, 18 files |

The size stays small because the face model isn't in the APK: Google Play Services delivers it and keeps it updated. What does ship is CameraX plus about 40 KB of SDK code.

### Requirements for the host app

- Flutter 3.24 or newer (Dart 3.5+), Android only.
- Android `minSdk` 24 or higher and `compileSdk` 35 or higher.
- **Google Play Services** on the phone, for the face model.
- The app requests the **CAMERA** permission before calling `LsnCapture.start`. The SDK declares it in its manifest, but the runtime prompt belongs to the app.

### Install: pick one

**1. Copy the folder:** unzip `dist/lsn_capture_sdk-1.6.3.zip` into your project, then:
```yaml
dependencies:
  lsn_capture_sdk:
    path: ./lsn_capture_sdk
```

**2. From this git repo:** once it's pushed, no copying is needed:
```yaml
dependencies:
  lsn_capture_sdk:
    git:
      url: https://github.com/akshat-sachan-LSN/zepiris.git
      ref: dress-code-poc        # or a release tag, e.g. lsn_capture_sdk-v1.6.3
      path: sdk/lsn_capture_sdk
```

Then `flutter pub get`, and:
```dart
import 'package:lsn_capture_sdk/lsn_capture_sdk.dart';

LsnCapture.warmUp();                       // early: face model + camera ready
final r = await LsnCapture.start(context, const LsnCaptureConfig(
  apiBase: 'https://3-108-193-187.sslip.io',
  checks: ['face_match', 'dress_color', 'logo'],
  sourceSelfieS3: 'https://bucket.s3.amazonaws.com/rider.jpg',
  challenge: 'blink',
));
print(r?.isCleared);
```

The LSN Checkpoint app (`mobile/lsn_checkpoint`) is a working example: `lib/screens/config_screen.dart`, `_openSdk`.

### Rebuild the zip

```bash
cd sdk && rm -f dist/lsn_capture_sdk-1.6.3.zip && zip -rq dist/lsn_capture_sdk-1.6.3.zip lsn_capture_sdk -x '*/build/*' '*/.dart_tool/*' '*.iml' '*/.idea/*' '*/.gradle/*' '*/local.properties'
```

## React Native (`react-native-lsn-capture/`, `@loadshare/rn-lsn-capture` v1.0.2)

The same native camera screen (the Kotlin `CaptureActivity` / `FaceEngine` / `Checks`, copied over) as an Android-only React Native module, plus the `/v1/checkpoint/score` call made from Kotlin. API: `isSupported`, `warmUp`, `capture`, `score`, `start`, `toScores`. Errors carry a `.code`. Full docs: [`react-native-lsn-capture/README.md`](react-native-lsn-capture/README.md).

Nothing is published (the package is `"private": true`). Build the tarball, then vendor it into the app:

```bash
cd sdk/react-native-lsn-capture && yarn install && yarn typescript && yarn test
yarn pack:dist                     # → sdk/dist/loadshare-rn-lsn-capture-1.0.2.tgz
# in the app: cp …/sdk/dist/loadshare-rn-lsn-capture-1.0.2.tgz vendor/
#             yarn add file:./vendor/loadshare-rn-lsn-capture-1.0.2.tgz
```

The host app requests the CAMERA runtime permission itself, and needs RN 0.72+ (old or new architecture), `minSdk` 24, `compileSdk` 35, Kotlin 2.0+ and Google Play Services on the phone. The returned JPEG is the host's to delete after upload. It is a native change: it needs a new store build, not CodePush.
