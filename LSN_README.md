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
| 5 | **Dress code, the decision** (is it the Loadshare uniform?) | **SigLIP2-base** image encoder + trained logistic-regression head | ML service, onnxruntime | `models/siglip2_base_vision.onnx` (built, not committed) + `zepiris/ml_inference/assets/dresscode_head.json` | Apache-2.0 (SigLIP2); head is ours | Yes, when the encoder file exists |
| 6 | Dress code, **colour score** | HSV "Loadshare blue" coverage of the torso (no weights) | ML service, OpenCV | `dresscode_detection.py` | own code | Always reported; decides only if #5 is missing |
| 7 | Dress code, **logo score** | Template match of the chest logo (no weights) | ML service, OpenCV | `zepiris/ml_inference/assets/loadshare_logo.png` | own code | Always reported; weak signal |
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
| `GET /ui` | 8 in the browser; calls `/v1/checkpoint/verify` |

Detection (#1) anchors everything: the face box locates the face for liveness and the torso for dress code. A face that fills the frame, cut off at the forehead or chin, is found with a padded retry.

---

## The checkpoint endpoint

`POST /v1/checkpoint/verify` has the same body as `/v1/faces/facematch/verify`. Each side is **either an image (base64 or `data:` URI) or an S3/HTTP link**:

```json
{
  "face_check_b64":   "<live selfie, base64>",
  "face_check_s3":    "https://… (instead of face_check_b64)",
  "source_selfie_b64":"<enrolled selfie, base64>",
  "source_selfie_s3": "https://… (instead of source_selfie_b64)",
  "threshold": 0.5,
  "dresscode_threshold": 0.35
}
```

`threshold` and `dresscode_threshold` are optional. The response looks like this:

```json
{
  "requestId": "…",
  "isCleared": true,
  "scores": {
    "faceMatch": 0.95, "faceThreshold": 0.5, "liveness": 0.99,
    "uniform": 0.91, "dressColour": 0.66, "logo": 0.50,
    "dresscode": 0.91, "dresscodeThreshold": 0.35
  },
  "faceMatch": { "…same block as /facematch/verify…" },
  "dresscode": { "isMatch": true, "engine": "siglip2", "…": "…" }
}
```

- `isCleared` is true only when the face matches (and passes liveness, if on) **and** the uniform matches.
- `scores.uniform` is the trained model's decision score. `dressColour` and `logo` are supporting signals.
- `dresscode.engine` is `"siglip2"` when the trained model decided, and `"hsv"` when the service fell back to the colour/logo rule.

---

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

## Configuration added by LSN

| Variable | Service | Default | Meaning |
|---|---|---|---|
| `ML_SERVICE_LIVENESS_ENABLED` | ML | `false` | Load MiniFASNet even in face-match-only mode |
| `ML_SERVICE_SPOOF_MIN_FACE_PX` | ML | `32` | Faces smaller than this are not liveness-scored (`reason: face_too_small`) |
| `ZEPIRIS_LIVENESS_ENABLED` | API | `false` | Gate `/facematch/verify` and `/checkpoint/verify` on liveness. Fails closed (503) if the model is missing |
| `ML_SERVICE_DRESSCODE_CLASSIFIER_ENABLED` | ML | `true` | Use the SigLIP2 uniform classifier when the encoder exists |
| `ML_SERVICE_DRESSCODE_ENCODER_PATH` | ML | `/app/models/siglip2_base_vision.onnx` | Falls back to `./models/<name>` on native runs |
| `ML_SERVICE_DRESSCODE_HEAD_PATH` | ML | *(bundled asset)* | Override the trained head JSON |
| `ZEPIRIS_DRESSCODE_THRESHOLD` | API | *(unset)* | Unset uses the engine's calibrated threshold: 0.35 for the model, 0.55 for the rule |

## Run locally

```bash
ML_SERVICE_LIVENESS_ENABLED=true ZEPIRIS_LIVENESS_ENABLED=true ./run_local.sh
```

Then open **http://localhost:8000/ui**. The camera needs `localhost` or https.

## Measured on this branch (Mac CPU, local)

- **Face match, 20 rider photos, all 190 pairs:** the same rider scored 0.97–1.00; different riders scored ≤ 0.33, against a 0.5 threshold. No errors.
- **Frame-filling close-ups:** these match (0.76). Liveness and dress code now get a real face box through the padded retry.
- **Checkpoint latency:** about 0.4–0.6 s per call on CPU (face match + liveness + dress code at the same time). Load-test on the GPU host before production.
