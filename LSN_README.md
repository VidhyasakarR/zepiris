# ZepIris at Loadshare (LSN)

Loadshare's fork of [zepto-labs/zepiris](https://github.com/zepto-labs/zepiris), extended for rider checkpoints:
**is this the enrolled rider, are they live, and are they wearing the Loadshare uniform?**

This file covers what LSN added and which models do what. The upstream [README.md](README.md) still describes the base service.

---

## Models in use

| # | Purpose | Model | Runs where | File | Licence | On by default? |
|---|---|---|---|---|---|---|
| 1 | **Face detection** (find the face) | SCRFD `det_500m` (InsightFace `buffalo_s`) | ML service, onnxruntime | downloaded to `~/.insightface/models/` | ⚠️ InsightFace weights: **non-commercial research only** | Yes (`face_tier=balanced`) |
| 2 | **Face match** (1:1 identity) | ArcFace `w600k_r50` (InsightFace `buffalo_l`), 512-d, cosine | ML service, onnxruntime | downloaded to `~/.insightface/models/` | ⚠️ **non-commercial research only** | Yes |
| 3 | **Liveness / anti-spoof** (print, photo-of-screen) | MiniFASNet V2 (2.7× crop) + V1SE (4.0× crop), averaged | ML service, onnxruntime | `models/minifasnet_v2_yakhyo.onnx`, `models/minifasnet_v1se_yakhyo.onnx` | Apache-2.0 | Off. Needs `ML_SERVICE_LIVENESS_ENABLED` + `ZEPIRIS_LIVENESS_ENABLED` |
| 4 | Liveness, screen-replay cue | Moiré / glare detector (FFT + highlights, no weights) | ML service, OpenCV | `zepiris/ml_inference/moire_detection.py` | own code | With liveness |
| 5 | **Dress code: three trained heads** on one encoder pass: `dress_color` (shirt is Loadshare blue), `logo` (Loadshare print present), `uniform` (both) | **SigLIP2-base** image encoder + three logistic-regression heads | ML service, onnxruntime | `models/siglip2_base_vision.onnx` (built, not committed) + `zepiris/ml_inference/assets/{dress_color,logo,dresscode}_head.json` | Apache-2.0 (SigLIP2); heads are ours | Yes, when the encoder file exists |
| 6 | Dress code, raw colour signal | HSV "Loadshare blue" coverage of the torso (no weights) | ML service, OpenCV | `dresscode_detection.py` | own code | Reported only (`blueCoverage`); decides only if #5 is missing |
| 7 | Dress code, raw logo signal | Template match of the chest logo (no weights) | ML service, OpenCV | `zepiris/ml_inference/assets/loadshare_logo.png` | own code | Reported only (`logoMatch`); cannot tell logo from no logo, so never decides |
| 8 | **Capture guide** (half-body skeleton in the UI) | MediaPipe Pose Landmarker lite | Browser (WASM/WebGL) | loaded from CDN at runtime | Apache-2.0 | Yes, in `/ui` |
| 9 | Legacy liveness fallback | MobileNetV3-Large binary classifier | ML service, PyTorch | `models/spoof_model.pth` | ZepIris-provided | Only if the MiniFASNet files are missing |
| 10 | NSFW check (full IQA only) | MobileNetV2 binary classifier | ML service, PyTorch | `models/nsfw_model.pth` | ZepIris-provided | Off in `face_match_only` mode |
| 11 | Blur check (full IQA only) | ResNet-18 binary classifier | ML service, PyTorch | `models/blur_model.pth` | ZepIris-provided | Off in `face_match_only` mode |

> ⚠️ **Licence risk (rows 1–2).** InsightFace's pretrained packs (`buffalo_l`, `buffalo_s`) are licensed for non-commercial research only. Upstream zepiris switched its default to **AuraFace** (`fal/AuraFace-v1`, Apache-2.0) for this reason. Port that as a face-model option, and re-check match thresholds, before relying on this commercially.

### Which models each endpoint uses

| Endpoint | Models |
|---|---|
| `POST /v1/checkpoint/verify` | 1, 2 (face match); 1, 3, 4 (liveness, if on); 1, 5, 6, 7 (dress code). All run at the same time |
| `POST /v1/faces/facematch/verify` | 1, 2; plus 1, 3, 4 when liveness is on |
| `POST /v1/dresscode/match` | 1, 5, 6, 7 |
| `GET /ui` | Operator page: 8 in the browser; calls `/v1/checkpoint/verify` |
| `POST /ui/selfie` | Rider selfie page (full screen): 8 in the browser; calls `/v1/checkpoint/verify` with the POSTed config |

Detection (#1) anchors everything: the face box locates the face for liveness and the torso for dress code. A face that fills the frame, cut off at the forehead or chin, is found with a padded retry.

---

## The checkpoint endpoint: the caller picks the checks

`POST /v1/checkpoint/verify` accepts a `checks` list naming what to verify on the live selfie:

| Check | What passes it | Needs `source_selfie_*`? |
|---|---|---|
| `face_match` | Same person as the enrolled selfie (and live, when `ZEPIRIS_LIVENESS_ENABLED`) | **yes** |
| `dress_color` | Shirt is Loadshare blue | no |
| `logo` | Loadshare print (chest logo or wordmark) is on the shirt | no |

Rules:
- **Only the listed checks run.** Face-only never calls the dress-code model; dress-only needs no source and never runs face match.
- `checks` omitted means all three. `[]` or an unknown name returns 422.
- **`isCleared` is true only when every requested check passes.**
- Each image is **either** base64 (`*_b64`, bare or `data:` URI) **or** a link (`*_s3`), never both.
- Threshold overrides (`threshold` for face, `dress_color_threshold`, `logo_threshold`) are refused with 403 unless `ZEPIRIS_ALLOW_THRESHOLD_OVERRIDE=true`, which is for testing only. They are always limited to 0.05–0.99.

### Examples

All three checks: selfie as base64, enrolled selfie as an S3 link.
```bash
curl -s -X POST http://localhost:8000/v1/checkpoint/verify -H 'Content-Type: application/json' -d '{"face_check_b64":"<BASE64_SELFIE>","source_selfie_s3":"https://bucket.s3.amazonaws.com/rider-123.jpg","checks":["face_match","dress_color","logo"]}'
```

Face match + dress colour only (logo ignored).
```bash
curl -s -X POST http://localhost:8000/v1/checkpoint/verify -H 'Content-Type: application/json' -d '{"face_check_s3":"https://bucket.s3.amazonaws.com/live.jpg","source_selfie_s3":"https://bucket.s3.amazonaws.com/rider-123.jpg","checks":["face_match","dress_color"]}'
```

Face match only.
```bash
curl -s -X POST http://localhost:8000/v1/checkpoint/verify -H 'Content-Type: application/json' -d '{"face_check_b64":"<BASE64_SELFIE>","source_selfie_b64":"<BASE64_ENROLLED>","checks":["face_match"]}'
```

Dress colour + logo only (no source needed).
```bash
curl -s -X POST http://localhost:8000/v1/checkpoint/verify -H 'Content-Type: application/json' -d '{"face_check_s3":"https://bucket.s3.amazonaws.com/live.jpg","checks":["dress_color","logo"]}'
```

From image files on disk (Linux `base64 -w0`; on macOS use `base64 -i file`):
```bash
printf '{"face_check_b64":"%s","source_selfie_b64":"%s","checks":["face_match","dress_color","logo"]}' "$(base64 -w0 selfie.jpg)" "$(base64 -w0 enrolled.jpg)" > body.json && curl -s -X POST http://localhost:8000/v1/checkpoint/verify -H 'Content-Type: application/json' -d @body.json
```

### Response
```json
{
  "requestId": "…",
  "checksRequested": ["face_match", "dress_color", "logo"],
  "isCleared": false,
  "checks": {
    "face_match":  {"passed": true,  "score": 0.998, "threshold": 0.5, "liveness": 0.99, "livenessFailed": false, "faceDetected": true},
    "dress_color": {"passed": true,  "score": 0.985, "threshold": 0.38},
    "logo":        {"passed": false, "score": 0.032, "threshold": 0.40}
  },
  "faceMatch": { "…full /facematch/verify block, only when face_match ran…" },
  "dresscode": { "…full /dresscode/match block, only when a dress check ran…" }
}
```

`checks` holds only the checks that were requested. Errors:

| Status | Cause |
|---|---|
| 400 | `face_match` requested without `source_selfie_*`, or both `_b64` and `_s3` given for one image |
| 422 | Empty or unknown `checks` |
| 503 | Dress check requested but the SigLIP2 model isn't loaded on the ML service, or liveness is on but its model is missing |

## Dress-code model: how it was chosen and trained

- **Candidates compared** (commercial-OK licences only): the current HSV + logo rule; SigLIP2-base zero-shot; SigLIP2-base + trained head; DINOv2-small + head; SigLIP2 + DINOv2.
- **Data:** 20 uniform photos of **18 riders** in two shirt designs (big vertical LOADSHARE wordmark; small chest logo), each augmented 20× for lighting, crop, mirroring and JPEG quality.
  - **Negatives:** 6 real photos of people not in uniform, plus hard negatives made from every uniform photo: the shirt recoloured red, green, grey or black, and all print painted out ("plain blue polo").
- **Evaluation:** 5-fold cross-validation **grouped by rider**, so no rider appears in both training and test.

| Model | Real uniform accepted | Plain blue, no logo, rejected | Wrong colour rejected | AUC |
|---|---|---|---|---|
| HSV + logo rule | 67% | 37% | 100% | 0.897 |
| **SigLIP2 + head (chosen)** | **98%** | **100%** | **100%** | **0.999** |
| DINOv2 + head | 84% | 98% | 83–100% | 0.965 |
| SigLIP2 + DINOv2 | 97% | 100% | 100% | 0.999 (no gain, 2× compute) |

- **Per-check heads (same data, same rider-grouped test):** `dress_color` scored AUC 1.000 and 99.7% balanced accuracy at threshold 0.38, beating the HSV coverage (AUC 0.989). `logo` scored AUC 0.999 and 98.7% at threshold 0.40. The logo head saw the print on red, green, grey and black shirts and plain blue shirts without it, so it learned the print independently of the colour.
- **Operating threshold:** 0.35, chosen on held-out scores. At that setting 99.5% of real uniforms are accepted and 99.7% of non-uniform images are rejected.
- **Input views:** the whole frame plus a face-anchored chest-to-stomach crop, embedded in one batch and concatenated (2 × 768 → head).
- **Preprocessing must be PIL bilinear 224×224 scaled to [-1, 1].** cv2 resizing shifted the embeddings to cosine ≈ 0.91 of the trained ones.
- **Caveat:** most negatives are synthetic. Before trusting the numbers, collect 30–50 real hard negatives: plain blue polos, other companies' blue uniforms, and people in the same warehouse not in uniform. Then retrain.

### Build the encoder file (one-time, 372 MB)

```bash
pip install "transformers>=4.50"
```

```bash
python scripts/export_dresscode_model.py
```

This writes `models/siglip2_base_vision.onnx`, which is gitignored. Bake it into the ML image. If it's missing, the service logs a warning and uses the colour/logo rule.

---

## Logo check: reading LOADSHARE on the shirt

The `logo` check is decided by OCR (PP-OCR through RapidOCR, ONNX on CPU, Apache-2.0, 15 MB of models inside the wheel), not by the learned logo head. That head had only ever seen "Loadshare print" against "no print", so it learned "white print on the chest", and a marathon tee passed.

- **Letters:** a line read on the T-shirt counts if its letters are a run of LOADSHARE of 4 or more letters, forwards or backwards: `LOADSHARE`, `SHARE`, `LOAD`, `ADSHARE`, `ERAHS`. Shorter reads must be exact, so a partly read `ROADSTAR` ("ROADS") fails. Reads of 7 or more letters may have one wrong or extra letter at the very start or end (`LOADSHARC`), which is where OCR misreads land. A wrong letter mid-word is another word (`LOADSTAR`, `OADSTAR`). Doubled letters are collapsed first (`LLOAD`), since LOADSHARE has none. An F counts as an E (`HARF`), since LOADSHARE has no F.
- **Photos cut at the chest:** vertical text that runs into the photo's top or bottom edge is a sliced wordmark. It needs only 3 letters (`ARE`, `LOA`), and 4+ letters may have one wrong letter at an end, the sliced one (`IARE`). On the 17 uniform photos cropped at chest height: 17/17 pass when the photo ends at the stomach (3 face heights below the top of the head), 15/17 mid-chest (2.5), and 8/17 just below the collar (2.0), where often only 2 letters show. Other words cut the same way still fail. The exception is a word one end letter away from a run, such as "CARE" for HARE.
- **On blue:** at least 45% of the fabric around the letters must be Loadshare blue. A shirt region with almost no blue is rejected without running OCR at all.
- **On the shirt:** only the area from the chin down is read (in face-box units: ±2.6 face widths, 0.9 to 7 face heights), so background signs don't count. If nothing matches, the region is read again mirrored (selfies saved mirrored by a camera app) and upside down.
- **Result:** `checks.logo.read` in the checkpoint response (`logoText` on `/v1/dresscode/match`) shows what was read, the run it matched, how much blue surrounds it, and the reason for a failure (`no_loadshare_text`, `not_on_blue`, `ocr_error`). An OCR error fails closed.
- **Measured:** 152 of 153 test images were decided correctly. The set was the 17 uniform photos plus 3 references, and copies with the print replaced by other brands (Swiggy, Zomato, Zepto, Blinkit, Amazon, Rapido…), other prints and icons, or re-coloured red, black or grey. The one miss is a 170×296 reference photo whose only mark is the small chest logo.
- **Cost:** about 0.4 to 0.7 s per image on a Mac CPU, more when the retries run.
- **Install:** `pip install --no-deps rapidocr_onnxruntime==1.4.4 pyclipper shapely`. Use `--no-deps` because it asks for the GUI `opencv-python`, which clashes with `opencv-python-headless`. The Dockerfile does this. Set `ML_SERVICE_DRESSCODE_LOGO_OCR_ENABLED=false` to go back to the learned head.

## Rider selfie page (`POST /ui/selfie`)

A full-screen camera page for riders. It is opened with a **POST**, as a form or JSON, whose body carries the checkpoint config. Nothing goes in the URL and nothing is configurable on screen.

| Field | Meaning |
|---|---|
| `checks` | Comma list or array of `face_match`, `dress_color`, `logo` (default: all) |
| `source_selfie_s3` / `source_selfie_b64` | The enrolled selfie; one of them is required for `face_match` |
| `zoom` | Camera zoom-out, `0.25` / `0.5` / `0.75` / `0.9` / `1` (default `0.5`). Chips on screen let the rider change it |
| `challenge` | Liveness challenge before Capture unlocks: `blink` (default), `turn`, `random` or `none` |
| `threshold`, `dress_color_threshold`, `logo_threshold` | Pass-mark overrides. **Refused (403) unless `ZEPIRIS_ALLOW_THRESHOLD_OVERRIDE=true`**, and always limited to 0.05–0.99 |

The page only talks to the server that served it; there is no API-address parameter, so another website can't point it elsewhere.

How the page behaves:
- **No auto-capture.** The outline turns green when the face is in the oval and the T-shirt is framed; only then is **Capture** enabled.
- **Live face checks before Capture (MediaPipe Face Landmarker, on the phone).** In priority order, the rider sees:

  | Condition | Prompt |
  |---|---|
  | No face | "Look at the camera" |
  | More than one face | "Only one person in the frame" |
  | Face overexposed (luma above 225) | "Too bright" |
  | Scene bright, face dark | "Light is behind you" |
  | Head turned | "Look straight at the camera" |
  | Head tilted | "Keep your head level" |
  | Eyes closed | "Open your eyes" |
  | Moving | "Hold still" |

  Calibration: on 18 real rider selfies, yaw stayed within ±0.08 (limit 0.11), pitch within ±0.09 (limit 0.12), and eye-closed peaked at 0.25 (limit 0.5).
- **Liveness challenge.** Once the rider is framed, the page asks for a blink, or a head turn and back. Capture stays locked until it is done, and it resets if the face leaves the frame. A printed photo or a still picture can't pass it.
  - It is a barrier on the phone, not proof. Server-side passive liveness (MiniFASNet, `ZEPIRIS_LIVENESS_ENABLED`) still decides.
  - **If the face models can't load (CDN blocked, bad network), the page fails closed.** It shows "Couldn't load the face check" with a retry, instead of skipping the challenge.
  - Two faces in the frame reset the challenge, so it can't be handed from one person to another.
- **Blur check right after capture.** The page measures sharpness on the face using the Crété-Roffet blur metric, in the browser, instantly. If it scores over **0.65**, it shows **"Photo is blurry — retake it, or the checks may fail"**, with Retake as the main button and "Submit anyway" as the other.
  - **Calibration:** the 19 genuine rider selfies score 0.31–0.48.
  - **Live test:** the warning started one blur step before the logo check began failing, and every capture that went on to fail had been warned. The logo is small print, so it's the first check to break on a soft photo.
- **Low light.** While the camera runs, the page measures the face's brightness. Below a mean luma of 90 it shows "💡 Low light — the screen will flash", and on Capture the whole screen turns white for 0.8–1.8 s to light the face. It waits for the camera to adjust its exposure and keeps the brightest frame.
  - A photo that's still dim (face under 110) is gamma-brightened before sending.
  - A face under 45 gets **"Too dark to see your face — move to a brighter place and retake"**. The blur check isn't trusted there, because in the dark it reads sensor noise as detail.
  - The Android app also turns screen brightness to full while the selfie page is open, so the flash really lights the face.
- **Burst capture.** Capture takes 5 frames in about 0.3 s and keeps the sharpest face, since motion blur varies frame to frame. In low light it keeps the sharpest of the brightest frames.
- **Capture quality check (`POST /v1/quality/check`).** This runs right after capture, using the ResNet-18 blur model on the face and on the T-shirt. The rider sees warnings such as "Face is blurry", "T-shirt is blurry", "T-shirt not visible", "Too dark" or "Face too far", with Retake as the main button and "Submit anyway" as the other.
  - **Face blur (threshold 0.5):** it caught 98% of blurred faces, including 94% in low light, while flagging 8% of sharp ones (AUC 0.994). This was measured on rider photos with focus, motion and defocus blur, half of them also dark and noisy.
  - **T-shirt blur (threshold 0.95):** it caught 75% with 2% false flags. The bar is stricter because plain fabric looks soft to the model.
  - **The old in-browser metric** had rated the real blurry dark captures "sharp". It remains only as a fallback when the server can't be reached; the check times out after 6 s.
  - **Light is judged on the photo as the camera took it.** The page sends the pre-brightening image here; judged after brightening, every dark photo would pass. A face overexposed above 230 gets "Too bright".
- **Robustness.**
  - Switching apps or locking the screen releases the camera, and it reopens when the rider returns.
  - A camera that stops is reported with a way to restart it.
  - A change in camera resolution re-fits the outline.
  - A failed capture never leaves the white flash on screen.
  - Retake is disabled while Submit is running, and replies that arrive for an older photo are ignored.
  - Server text is always inserted as text, never as HTML.
- **Submit** sends the **captured photo** (`face_check_b64`) plus the config above in the body of `POST /v1/checkpoint/verify`, then shows the result.
- **Zoom:** a web page can't widen the lens. Filling a tall phone screen crops the camera picture, and zooming out crops less, showing more of the camera's real view. From about 0.6× down, the camera's whole picture is shown with bars above and below, so 0.5× and 0.25× look the same on most phones. If the front camera reports an optical zoom below 1, its widest setting is used. The guide, the skeleton and the saved photo all use the visible part of the picture.
- **Camera mode:** the page asks for 1440×1080 with `resizeMode: none`. Chrome on Android reads width and height in the sensor's landscape orientation, so this returns the native 3:4 mode (1080×1440 portrait), the full view the phone's own camera app shows. Asking for 1080×1440 made Chrome crop the frame to a landscape strip, a heavily zoomed-in picture (seen on a Redmi 2312FRAFDI, WebView 152).
- **Host apps:** the page posts the result to a `ZepirisBridge` JavaScript channel, which the Android app listens on.

```bash
curl -s -X POST https://<host>/ui/selfie -d 'checks=face_match,dress_color,logo' -d 'source_selfie_s3=https://bucket.s3.amazonaws.com/rider.jpg' -d 'zoom=0.5'
```

## Android app (`mobile/lsn_checkpoint`)

The flow is **Config → selfie page → result**.

1. **Config screen:** which checks to run, the camera size (zoom 0.25× / 0.5× / 0.75× / 0.9× / 1×, default 0.5×), the liveness check (blink / head turn / random / off), and the enrolled selfie (S3 link, or an image from the gallery, copied into app storage). These are saved on the phone.
2. **Open selfie:** loads the server's `/ui/selfie` in a full-screen WebView, POSTing that config. It's the same page as on the web, so the guide, zoom, green-to-capture and Submit behave identically.
3. **Result:** the page shows it, and the app shows a Cleared / Not cleared chip.

**Fast open (standby page).** The camera guide needs about 12 MB of models (MediaPipe pose, face and WebAssembly) that take seconds to download and compile.
- **Preload:** as soon as the Config screen shows, the app loads `GET /ui/selfie` in a background WebView. The WebView sits in a 1×1 box on screen, because a WebView must be attached to render and use the GPU. The page calls `preload()` in `guide.js`, which downloads, compiles and warms up both models.
- **Open:** Open selfie hands the page the rider's config with `window.zepirisConfigure()`. The page validates it through `POST /ui/selfie/config` (the same rules as `POST /ui/selfie`) while the camera is already opening.
- **Back:** going back calls `window.zepirisReset()`, which stops the camera and clears the rider's data but keeps the models in memory.
- **Measured on a Redmi (2312FRAFDI):** models ready about 4.5 s after the app opens, and the camera live about 1.5 s after tapping Open selfie. Before, the model load came after the tap.
- **Deadlines:** model setup gets 10 s on GPU and 20 s on CPU, so a WebView that is backgrounded mid-load cannot hang the camera screen.
- **Older servers:** if the server's page has no standby mode, the app falls back to POSTing the config.
- **Test builds:** `--dart-define=API_BASE=http://localhost:8000` (with `adb reverse tcp:8000 tcp:8000`) points the app at a laptop over USB, and `--dart-define=WEBVIEW_DEBUG=true` allows `chrome://inspect`.

Build:
```bash
cd mobile/lsn_checkpoint && flutter build apk --release --split-per-abi
```

This writes `build/app/outputs/flutter-apk/app-release.apk`. Install it:
```bash
adb install -r mobile/lsn_checkpoint/build/app/outputs/flutter-apk/app-release.apk
```

Notes:
- **The server address is built in** (`kDefaultApiBase` in `lib/settings.dart`, now `https://3-108-193-187.sslip.io`) and not shown to riders. It must be https, because WebView cameras only work on secure pages. On the EC2, Caddy terminates TLS (a Let's Encrypt certificate via sslip.io) and forwards to the API on 8000.
- The WebView uses a non-browser user agent so ngrok's free-tier "Visit site" warning page doesn't interrupt it.
- **WebView lock-down:**
  - Only the configured server's pages get the camera (never the microphone).
  - Only its `/ui/selfie` page may report a result.
  - Links and redirects to any other host are blocked.
  - Server errors on opening the page (403/413/422/5xx) show a readable message.
  - When the app goes to the background, the page is told to release the camera, and it reopens on return.
- **The Cleared / Not cleared chip is a display.** The checkpoint decision is the server's response to `/v1/checkpoint/verify`. Anything that grants access should use that response, not the chip.
- The release build is signed with the debug key; set up a real signing key before distributing.

## Configuration added by LSN

| Variable | Service | Default | Meaning |
|---|---|---|---|
| `ML_SERVICE_LIVENESS_ENABLED` | ML | `false` | Load MiniFASNet even in face-match-only mode |
| `ML_SERVICE_SPOOF_MIN_FACE_PX` | ML | `32` | Faces smaller than this are not liveness-scored (`reason: face_too_small`) |
| `ZEPIRIS_LIVENESS_ENABLED` | API | `false` | Gate `/facematch/verify` and `/checkpoint/verify` on liveness. Fails closed (503) if the model is missing |
| `ML_SERVICE_DRESSCODE_CLASSIFIER_ENABLED` | ML | `true` | Use the SigLIP2 uniform classifier when the encoder exists |
| `ML_SERVICE_DRESSCODE_ENCODER_PATH` | ML | `/app/models/siglip2_base_vision.onnx` | Falls back to `./models/<name>` on native runs |
| `ML_SERVICE_DRESSCODE_LOGO_OCR_ENABLED` | ML | `true` | Decide the logo check by reading LOADSHARE on blue fabric (see above) |
| `ML_SERVICE_DRESSCODE_HEAD_PATH` | ML | *(bundled asset)* | Override the trained head JSON |
| `ML_SERVICE_QUALITY_CHECK_ENABLED` | ML | `true` | Load the blur model (`models/blur_model.pth`) for `/v1/quality/check`, even in face-match-only mode |
| `ZEPIRIS_ALLOW_THRESHOLD_OVERRIDE` | API | `false` | Accept per-request pass marks on `/v1/checkpoint/verify` and `/ui/selfie`. Keep it off in production: a device could otherwise clear itself |
| `ZEPIRIS_IMAGE_URL_BLOCK_PRIVATE` | API | `true` | Refuse image URLs (`*_s3`) that resolve to loopback / private / link-local addresses (the ML service, cloud metadata `169.254.169.254`, the VPC). Checked on every redirect hop. Set `false` only for local testing against a LAN file server |
| `ZEPIRIS_IMAGE_URL_ALLOWED_HOSTS` | API | *(empty = any public host)* | Comma list of allowed image hosts, exact or `.suffix`, e.g. `.amazonaws.com`. Recommended in production |
| `ZEPIRIS_DRESSCODE_THRESHOLD` | API | *(unset)* | Unset uses the engine's calibrated threshold: 0.35 for the model, 0.55 for the rule |

Images over 50 megapixels are refused with 413 before they are decoded. That includes images past Pillow's own bomb limit, and files whose size can't be read are refused rather than passed to OpenCV. This stops a small, highly compressed file from expanding to gigabytes in memory. Decoding runs off the event loop. Image URLs are downloaded as a stream and cut off at 5 MB.

Every `threshold` field is limited to 0.05–0.99 and must be finite, on the older `/v1/faces/*` and `/v1/dresscode/match` endpoints too. A negative threshold would otherwise pass anything.

## Run locally

```bash
ML_SERVICE_LIVENESS_ENABLED=true ZEPIRIS_LIVENESS_ENABLED=true ./run_local.sh
```

Then open **http://localhost:8000/ui**. The camera needs `localhost` or https.

## Measured on this branch (Mac CPU, local)

- **Face match, 20 rider photos, all 190 pairs:** the same rider scored 0.97–1.00; different riders scored ≤ 0.33, against a 0.5 threshold. No errors.
- **Frame-filling close-ups:** these match (0.76). Liveness and dress code now get a real face box through the padded retry.
- **Checkpoint latency:** about 0.4–0.6 s per call on CPU (face match + liveness + dress code at the same time). Load-test on the GPU host before production.
