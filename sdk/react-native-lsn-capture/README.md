# @loadshare/rn-lsn-capture

React Native SDK for the Zepiris LSN checkpoint selfie. It opens a **native Android camera screen** (oval guide, live face and T-shirt checks, blink / head-turn liveness), returns the JPEG, and can score it on the Zepiris server (`POST /v1/checkpoint/score`: face match, liveness, dress colour, logo).

It is a port of the Flutter plugin in `../lsn_capture_sdk` (same camera screen, same score request). **Android only**: on iOS every call rejects with `unsupported_platform` and `isSupported()` returns `false`.

## Requirements (host app)

- React Native 0.72+ (built and tested against 0.77.3). It is a legacy native module, so it also works in the new architecture through the interop layer.
- Android `minSdk` 24+, `compileSdk` 35+, Kotlin 2.0+, Java 17. The module reads `compileSdkVersion`, `minSdkVersion`, `targetSdkVersion` and `kotlinVersion` from the root project's `ext`.
- **Google Play Services** on the phone (it delivers the ML Kit face model).
- **The host requests the CAMERA runtime permission before `capture()` / `start()`.** The SDK declares it in its manifest, but if it has not been granted the call rejects with `no_permission`.
- Brings in CameraX 1.4.2. If the app has its own direct CameraX dependencies, put them on the same version.

## Install (vendored tarball, nothing is published)

The package is `"private": true`, so it can't be published by mistake. Build the tarball in this repo:

```bash
cd zepiris/sdk/react-native-lsn-capture
yarn install && yarn typescript && yarn test
yarn pack:dist            # → zepiris/sdk/dist/loadshare-rn-lsn-capture-<ver>.tgz
```

Copy it into the app and install it with `file:`:

```bash
mkdir -p vendor && cp <zepiris>/sdk/dist/loadshare-rn-lsn-capture-1.0.0.tgz vendor/
yarn add file:./vendor/loadshare-rn-lsn-capture-1.0.0.tgz
npx react-native config | grep -i lsn    # autolinking picked up LsnCapturePackage
```

Commit `vendor/*.tgz`, `package.json` and `yarn.lock`. **To update:** bump `version`, run `yarn pack:dist`, replace the tgz in `vendor/`, then run `yarn add` again. Yarn 1 caches `file:` tarballs by name, so always change the version (or run `yarn cache clean @loadshare/rn-lsn-capture`). This is a native change, so it needs a new store build: it cannot ship over CodePush.

## Usage

```ts
import { PermissionsAndroid } from 'react-native';
import * as LsnCapture from '@loadshare/rn-lsn-capture';

if (LsnCapture.isSupported()) LsnCapture.warmUp(); // early, e.g. on the home screen

async function goOnlineCheck(referenceSelfieUrl: string) {
  const granted = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.CAMERA);
  if (granted !== PermissionsAndroid.RESULTS.GRANTED) return;
  try {
    // 1. photo only (e.g. to upload it and score it in parallel)
    const shot = await LsnCapture.capture({ challenge: 'blink' });
    const scores = await LsnCapture.score({
      apiBase: 'https://3-108-193-187.sslip.io',
      checks: ['face_match', 'dress_color', 'logo'],
      faceCheckPath: shot.path,
      sourceSelfieS3: referenceSelfieUrl,
    });
    console.log(shot.challengePassed, scores.faceSimilarity, scores.dressColor, scores.logo);

    // 2. or both in one call
    const { capture, scores: s } = await LsnCapture.start({
      challenge: 'random',
      apiBase: 'https://3-108-193-187.sslip.io',
      sourceSelfieS3: referenceSelfieUrl,
    });
  } catch (e: any) {
    if (e.code === 'cancelled') return; // the rider backed out
    console.warn(e.code, e.message);
  }
}
```

The JPEG stays in the app cache (`capture.path` / `capture.uri`) so the host can upload it. The SDK deletes its own photos that are more than 10 minutes old the next time the camera opens.

## API

| Function | Returns |
|---|---|
| `isSupported()` | `true` on Android when the native module is linked |
| `warmUp()` | `{ modelReady, ms, reason }`: loads the face model and CameraX |
| `capture(opts?)` | `{ path, uri, challenge, challengePassed, stats }` |
| `score(opts)` | `LsnScores` (raw JSON in `.raw`) |
| `start(opts)` | `{ capture, scores }`: the score options are validated before the camera opens |
| `toScores(raw)` | pure mapper from the raw score JSON to `LsnScores` |

**`capture` options:** `challenge` (`'none' \| 'blink' \| 'turn' \| 'random'`, default `'blink'`), `light` (start with the screen light on), `maxSide` (long side in px, default 2592, 0 = full resolution), `jpegQuality` (default 92), `brightness` (starting EV, −2..+2).

**`score` options:** `apiBase` (http(s), trailing `/` dropped), `checks` (non-empty subset of `face_match`, `dress_color`, `logo`; default all three), exactly one of `faceCheckPath` (a local path or `file://` URI, base64-encoded natively) or `faceCheckS3`, and `sourceSelfieS3` or `sourceSelfieB64` (required when `face_match` is requested). Also `timeoutMs` (default 40000) and `headers`.

**`LsnScores`:** `requestId`, `scoredAt`, `checksRequested`, `faceSimilarity`, `liveness` (null while server liveness is off), `faceDetected`, `dressColor`, `logo`, `logoRead` (`{text, matched, onBlue, reason}`), `imageSha256`, `raw`. These are scores only: the pass / fail decision belongs to the host backend.

## Errors

Every rejection is an `Error` with `.code`:

| code | when |
|---|---|
| `cancelled` | the rider backed out of the camera screen |
| `busy` | a capture is already running |
| `no_activity` | there is no foreground activity |
| `no_permission` | CAMERA permission has not been granted |
| `face_check_unavailable` | the Play Services face model did not load within 10 s |
| `capture_failed` | the camera screen or the camera failed |
| `invalid_config` | bad score options |
| `network` | the server could not be reached |
| `timeout` | there was no reply within `timeoutMs` |
| `server` | a non-2xx reply (the message comes from FastAPI's `detail`) or a reply without `scores` |
| `unsupported_platform` | not Android, or the native module is not linked |

## Layout

- `src/`: the TypeScript API (`index.ts`, `types.ts`, `scores.ts`, `errors.ts`, `NativeLsnCapture.ts`).
- `android/src/main/java/com/loadshare/lsncapture/`:
  - `CaptureActivity.kt`, `FaceEngine.kt`, `Checks.kt`: copied from the Flutter plugin, with only the package renamed.
  - `LsnCaptureModule.kt` and `LsnCapturePackage.kt`: the RN bridge.
  - `ScoreClient.kt`: the score call.
- `android/src/test/`: `ChecksTest.kt` (live checks + liveness) and `ScoreClientTest.kt` (validation + error parsing).
