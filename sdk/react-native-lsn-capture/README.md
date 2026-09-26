# @loadshare/rn-lsn-capture

React Native SDK for the Zepiris LSN checkpoint selfie. It opens a **native Android camera screen** (oval guide, live face and T-shirt checks, blink / head-turn liveness), returns the JPEG, and can score it on the Zepiris server (`POST /v1/checkpoint/score`: face match, liveness, dress colour, logo).

It is a port of the Flutter plugin in `../lsn_capture_sdk` (same camera screen, same score request). **Android only**: on iOS every call rejects with `unsupported_platform` and `isSupported()` returns `false`.

## Requirements (host app)

- React Native 0.72+ (built and tested against 0.77.3). It is a legacy native module, so it also works in the new architecture (bridgeless) through the interop layer.
- Android `minSdk` 24+, `compileSdk` 35+, Kotlin 2.0+, Java 17, AGP 8+. The module reads `compileSdkVersion`, `minSdkVersion`, `targetSdkVersion` and `kotlinVersion` from the root project's `ext`.
- **Google Play Services** on the phone (it delivers the ML Kit face model).
- **The host requests the CAMERA runtime permission before `capture()` / `start()`.** The SDK declares it in its manifest, but if it has not been granted the call rejects with `no_permission`. No storage permission is used.
- Brings in CameraX 1.4.2. Gradle uses the highest CameraX version in the app (titan-rider-app resolves 1.5.0-alpha03); if the app has its own direct CameraX dependencies, keep them on one version.
- Uses the app's OkHttp (React Native ships it); works with OkHttp 4.9+ and 5.x.
- Hosts that force `androidx.lifecycle:lifecycle-viewmodel` below 2.6 (titan-rider-app forces 2.5.1) are supported: `CaptureActivity` extends a small Java shim, `LsnBaseActivity`, that keeps it compiling against both ViewModel-owner API shapes. Don't remove it.
- Release builds: the SDK ships its own consumer R8 rules; nothing to add.

## Install (vendored tarball, nothing is published)

The package is `"private": true`, so it can't be published by mistake. Build the tarball in this repo:

```bash
cd zepiris/sdk/react-native-lsn-capture
yarn install && yarn typescript && yarn test
yarn pack:dist            # → zepiris/sdk/dist/loadshare-rn-lsn-capture-<ver>.tgz
```

Copy it into the app and install it with `file:`:

```bash
mkdir -p vendor && cp <zepiris>/sdk/dist/loadshare-rn-lsn-capture-1.0.2.tgz vendor/
yarn add file:./vendor/loadshare-rn-lsn-capture-1.0.2.tgz
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
  } catch (e) {
    const err = e as LsnCapture.LsnError;
    if (err.code === 'cancelled') return; // the rider backed out
    console.warn(err.code, err.message); // show riders your own copy per code
  }
}
```

The JPEG stays in the app's private cache (`capture.path` / `capture.uri`) so the host can upload it. **The host owns that file**: delete it once uploaded. The SDK only deletes its own photos (`lsn_capture_*.jpg`) that are more than 10 minutes old, the next time the camera opens; Android may also clear the cache under storage pressure.

## API

| Function | Returns |
|---|---|
| `isSupported()` | `true` on Android when the native module is linked. Never throws (`false` in Jest / Expo Go / iOS) |
| `warmUp()` | `{ modelReady, ms, reason }`: loads the face model and CameraX. Resolves with `modelReady: false` instead of rejecting; the first run can take long (Play Services downloads the model), so fire and forget it |
| `capture(opts?)` | `{ path, uri, challenge, challengePassed, stats }` |
| `score(opts)` | `LsnScores` (raw JSON in `.raw`) |
| `start(opts)` | `{ capture, scores }`: the score options are validated before the camera opens; any `faceCheckPath` / `faceCheckS3` passed is ignored (the new photo is scored) |
| `toScores(raw)` | pure mapper from the raw score JSON to `LsnScores`; never throws, wrongly-typed fields become `null` |

**`capture` options:** `challenge` (`'none' \| 'blink' \| 'turn' \| 'random'`, default `'blink'`; case-insensitive, an unknown value falls back to `'blink'`), `light` (start with the screen light on), `maxSide` (long side in px, default 2592, 0 = full resolution), `jpegQuality` (60-100, default 92), `brightness` (starting EV, −2..+2). Out-of-range numbers are clamped and non-numbers replaced by the default: `capture()` never rejects for its options. The photo is at most ~4.5 MB (the quality steps down to fit).

**`score` options:** `apiBase` (http(s), trailing `/` dropped, no query or fragment), `checks` (non-empty subset of `face_match`, `dress_color`, `logo`; default all three), exactly one of `faceCheckPath` (a local path or `file://` URI, at most 5 MiB, base64-encoded natively so it never crosses the bridge) or `faceCheckS3`, and `sourceSelfieS3` or `sourceSelfieB64` (required when `face_match` is requested; base64 at most 5 MiB decoded). Also `timeoutMs` (whole call, 1..600000, default 40000) and `headers` (extra headers such as a gateway token; printable-ASCII values, no CR/LF; `Content-Type`, `Content-Length`, `Host`, `Connection`, `Transfer-Encoding` are owned by the SDK and ignored). Bad options reject with `invalid_config` before any network call. The SDK sends `User-Agent: LSNCaptureSDK-RN/<version>` unless `headers` overrides it.

**HTTP vs HTTPS:** use `https://` in production (the request carries the rider's photo). `http://` is accepted for local testing, but Android blocks cleartext unless the app's network security config allows that host; then the call rejects with `invalid_config`. An `https` → `http` redirect is never followed.

**`LsnScores`:** `requestId`, `scoredAt`, `checksRequested`, `faceSimilarity`, `liveness` (null while server liveness is off), `faceDetected`, `dressColor`, `logo`, `logoRead` (`{text, matched, onBlue, reason}`), `imageSha256`, `raw`. These are scores only: the pass / fail decision belongs to the host backend.

### Threading and lifecycle

- All functions are safe to call from JS at any time. The camera screen is launched on the main thread; `score()` runs on a background thread (the promise resolves on it); nothing blocks the JS thread.
- One capture at a time: a second `capture()` while the screen is open rejects with `busy`. The pending capture always settles: with the result, `cancelled` (back button), or `capture_failed` if the screen closes without its result reaching the module (e.g. a host activity that does not forward `onActivityResult` to React Native), so the SDK can never get stuck on `busy`.
- If Android destroys and recreates the app's activity while the camera screen is open (low memory, "Don't keep activities"), the result is still delivered.
- If the **app process** is killed while the camera screen is open, the JS promise dies with the old JS runtime: nothing resolves, and the photo taken is orphaned (cleaned by the next capture). The app restarts from its launch screen as usual.
- A JS reload / React context teardown drops a pending capture silently (there is no JS left to reject to) and lets in-flight `score()` calls finish within their `timeoutMs`.

### Privacy

- The SDK never logs image bytes, base64, header values or URLs with tokens. Error messages don't echo header values.
- Photos are written only to the app's private `cacheDir`; the SDK needs no storage permission.
- `score()` uses its own OkHttp client (not the app's `fetch` client), so app-level network interceptors / body loggers never see the photo.

## Errors

Every rejection is an `Error` with `.code` (type `LsnError`). `message` is developer-facing English (it may include the server's `detail`): map `code` to your own rider-facing copy. If the native side ever rejects with a code outside this list, `code` is the function's fallback (`capture_failed` for `capture` / `warmUp`, `network` for `score`) and the original is kept in `.nativeCode`.

| code | when |
|---|---|
| `cancelled` | the rider backed out of the camera screen |
| `busy` | a capture is already running |
| `no_activity` | there is no foreground activity |
| `no_permission` | CAMERA permission has not been granted |
| `face_check_unavailable` | the Play Services face model did not load within 10 s |
| `capture_failed` | the camera screen or the camera failed, or the screen closed without returning a result |
| `invalid_config` | bad score options (see above), a missing / empty / over-5 MiB `faceCheckPath`, or cleartext `http://` blocked by the app |
| `network` | the server could not be reached |
| `timeout` | there was no complete reply within `timeoutMs` |
| `server` | a non-2xx reply (the message comes from FastAPI's `detail`, max 300 chars), a non-JSON reply, a reply without `scores`, or a reply over 1 MB |
| `unsupported_platform` | not Android, or the native module is not linked |

## Layout

- `src/`: the TypeScript API (`index.ts`, `types.ts`, `scores.ts`, `errors.ts`, `NativeLsnCapture.ts`).
- `android/src/main/java/com/loadshare/lsncapture/`:
  - `CaptureActivity.kt`, `FaceEngine.kt`, `Checks.kt`: copied from the Flutter plugin, with only the package renamed.
  - `LsnBaseActivity.java`: the ViewModel-owner shim `CaptureActivity` extends (see Requirements).
  - `LsnCaptureModule.kt` and `LsnCapturePackage.kt`: the RN bridge.
  - `ScoreClient.kt`: the score call (validation, body, OkHttp, reply parsing).
- `android/src/test/`: `ChecksTest.kt` (live checks + liveness) and `ScoreClientTest.kt` (validation, header sanitising, file / reply size limits, error parsing, and end-to-end calls against MockWebServer).
- `src/__tests__/`: Jest tests for `toScores`, error mapping, option validation and the JS API (native module mocked).
