ZepIris — Third-Party Notices
==================================

This file lists the third-party libraries used by ZepIris, along with
their respective licenses. These libraries are installed as dependencies
(via pip/Poetry) and bundled in Docker images.

ZepIris itself is licensed under the MIT License.

--------------------------------------------------------------------------------

Bundled model weights
----------------------

SigLIP2 image encoder (dress-code / uniform classifier)
  File: models/siglip2_base_vision.onnx (not committed; built by
        scripts/export_dresscode_model.py from google/siglip2-base-patch16-224)
  Only the vision tower is exported; a linear head trained on Loadshare rider
  photos (zepiris/ml_inference/assets/dresscode_head.json) sits on top.
  License: Apache-2.0
  URL: https://huggingface.co/google/siglip2-base-patch16-224

--------------------------------------------------------------------------------

MediaPipe Pose Landmarker (browser only, capture guide in /ui)
  Loaded at runtime from cdn.jsdelivr.net (@mediapipe/tasks-vision) and
  storage.googleapis.com (pose_landmarker_lite.task); not bundled.
  License: Apache-2.0
  URL: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

--------------------------------------------------------------------------------

MiniFASNet Anti-Spoofing (ONNX) — two-model ensemble
  Files: models/minifasnet_v2_yakhyo.onnx, models/minifasnet_v1se_yakhyo.onnx
  Source: github.com/yakhyo/face-anti-spoofing (release assets)
  ONNX exports of minivision-ai/Silent-Face-Anti-Spoofing (MiniFASNetV2 + V1SE).
  License: Apache-2.0
  URL: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  NOTE: these exports take RAW [0,255] input (the /255 scaling is in-graph) and
  use class index 1 = real; the two models are ensembled (averaged prob_live)
  with per-model crop scales (V2 2.7x, V1SE 4.0x). See onnx_spoof_detection.py.
  Also bundled: models/minifasnet_v2_spoof.onnx (garciafido HF export, used by
  tests as a single-model preprocessing guard).

--------------------------------------------------------------------------------

FastAPI
  Version range: >=0.127.0,<0.200.0
  License: MIT
  URL: https://github.com/fastapi/fastapi
  Copyright (c) 2018-present Sebastián Ramírez

Starlette
  Version range: >=0.49.1,<1.0.0 (direct; mitigates CVE-2025-62727 in FileResponse Range handling)
  License: BSD-3-Clause
  URL: https://github.com/encode/starlette
  Copyright (c) 2018-present Encode OSS Ltd.

Uvicorn
  Version range: >=0.32.0,<0.33.0
  License: BSD-3-Clause
  URL: https://github.com/encode/uvicorn
  Copyright (c) 2017-present Encode OSS Ltd.

Pydantic / pydantic-settings
  Version range: >=2.6.0,<3.0.0
  License: MIT
  URL: https://github.com/pydantic/pydantic
  Copyright (c) 2017-present Samuel Colvin

python-multipart
  Version range: >=0.0.26,<0.1.0
  License: Apache-2.0
  URL: https://github.com/Kludex/python-multipart
  Copyright (c) 2012-present Andrew Dunham, Marcelo Trylesinski

MinIO Python SDK
  Version range: >=7.2.0,<8.0.0
  License: Apache-2.0
  URL: https://github.com/minio/minio-py
  Copyright (c) 2015-present MinIO, Inc.

PyMilvus
  Version range: >=2.4.0,<3.0.0
  License: Apache-2.0
  URL: https://github.com/milvus-io/pymilvus
  Copyright (c) 2019-present Zilliz

OpenCV (opencv-python-headless)
  Version range: >=4.10.0,<5.0.0
  License: Apache-2.0
  URL: https://github.com/opencv/opencv-python
  Copyright (c) 2000-present Intel Corporation, Willow Garage Inc.,
  Itseez Inc., and other OpenCV contributors

NumPy
  Version range: >=1.26.0,<3.0.0
  License: BSD-3-Clause
  URL: https://github.com/numpy/numpy
  Copyright (c) 2005-present NumPy Developers

Pillow
  Version range: >=10.0.0,<12.0.0
  License: HPND (Historical Permission Notice and Disclaimer)
  URL: https://github.com/python-pillow/Pillow
  Copyright (c) 1995-2011 Fredrik Lundh and contributors,
  2010-present Jeffrey A. Clark and contributors

Hugging Face Hub
  Version range: >=0.16.0,<1.0.0
  License: Apache-2.0
  URL: https://github.com/huggingface/huggingface_hub
  Copyright (c) 2020-present Hugging Face, Inc.

HTTPX
  Version range: >=0.24.0,<1.0.0
  License: BSD-3-Clause
  URL: https://github.com/encode/httpx
  Copyright (c) 2019-present Encode OSS Ltd.

Requests
  Version range: >=2.33.0,<3.0.0 (direct; transitive from Hugging Face Hub and others)
  License: Apache-2.0
  URL: https://github.com/psf/requests
  Copyright (c) 2011-present Kenneth Reitz and contributors

--------------------------------------------------------------------------------

Optional ML Dependencies (installed with the [ml] extra)

PyTorch
  Version range: >=2.2.0
  License: BSD-3-Clause
  URL: https://github.com/pytorch/pytorch
  Copyright (c) 2016-present Facebook, Inc. (Meta Platforms, Inc.)

TorchVision
  Version range: >=0.17.0
  License: BSD-3-Clause
  URL: https://github.com/pytorch/vision
  Copyright (c) 2016-present Soumith Chintala

InsightFace
  Version range: >=0.7.3
  License: MIT (source code only)
  URL: https://github.com/deepinsight/insightface
  Copyright (c) 2018-present InsightFace contributors

  NOTE: Pre-trained model weights downloaded by this library are licensed for
  non-commercial research purposes only. Commercial use requires a separate
  license. Contact: recognition-oss-pack@insightface.ai

ONNX Runtime
  Version range: >=1.17.0
  License: MIT
  URL: https://github.com/microsoft/onnxruntime
  Copyright (c) 2018-present Microsoft Corporation

--------------------------------------------------------------------------------

Infrastructure (used via Docker Compose, not bundled in ZepIris images)

Milvus
  License: Apache-2.0
  URL: https://github.com/milvus-io/milvus
  Copyright (c) 2019-present Zilliz

MinIO
  License: GNU AGPL v3.0 (server), Apache-2.0 (client SDK)
  URL: https://github.com/minio/minio
  Copyright (c) 2015-present MinIO, Inc.

etcd
  License: Apache-2.0
  URL: https://github.com/etcd-io/etcd
  Copyright (c) 2013-present CoreOS, Inc. and etcd Authors

--------------------------------------------------------------------------------

Full license texts for each license type referenced above:

MIT License:
  https://opensource.org/licenses/MIT

BSD 3-Clause License:
  https://opensource.org/licenses/BSD-3-Clause

Apache License 2.0:
  https://www.apache.org/licenses/LICENSE-2.0

HPND License:
  https://opensource.org/licenses/HPND

GNU Affero General Public License v3.0:
  https://www.gnu.org/licenses/agpl-3.0.html
