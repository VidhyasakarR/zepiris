# ML inference service — PyTorch + InsightFace. Local dev: `poetry install --extras ml`.
# Build: docker build -f ml_inference.Dockerfile -t zepiris-ml-service .
#
# Runtime deps (including torch) come from poetry.lock. Place model weights in models/
# at build time (see README / ML_INFERENCE_SERVICE.md).

FROM python:3.12.3-slim

LABEL org.opencontainers.image.title="ZepIris ML Inference"
LABEL org.opencontainers.image.description="Face IQA, embedding, and detection microservice"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# OpenCV / InsightFace runtime libs + g++ for native wheels + curl for HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        g++ \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock poetry.toml README.md ./
COPY zepiris/ zepiris/

ARG POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && poetry install --only main --extras ml --no-root --no-ansi \
    && pip install --no-cache-dir . \
    && pip uninstall -y poetry

# Logo check OCR runs HERE, in the ML service (logo_text.py), not in the API.
# It used to be installed only in the API image, so every ML build since the
# logo-OCR rule landed logged "Logo OCR unavailable" and failed every logo check
# with reason ocr_unavailable. Same pins as the API Dockerfile. --no-deps: it
# would pull GUI opencv-python (clashes with headless cv2) and the CPU
# onnxruntime (the GPU block below must stay the only ORT). Before the g++ purge
# in case pyclipper/shapely need to build.
RUN pip install --no-cache-dir --no-deps "rapidocr_onnxruntime==1.4.4" \
    && pip install --no-cache-dir "pyclipper==1.4.0" "shapely==2.1.2" "PyYAML>=6" "six>=1.16" "tqdm>=4.66"

RUN apt-get update \
    && apt-get purge -y g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# GPU builds: swap the CPU wheel for onnxruntime-gpu plus the CUDA 12 / cuDNN 9
# libraries it links against (the [cuda,cudnn] extras pull them from PyPI, so the
# base image needs no CUDA toolkit — only the host driver, via the NVIDIA
# container runtime). Empty, the default, leaves the CPU wheel alone.
#
# The CPU wheel has no CUDAExecutionProvider, so a GPU instance built without
# this silently runs inference on the host cores. See docs/CPU_VS_GPU.md §5.2.
#
# cuDNN is pinned because the [cudnn] extra only asks for ~=9.0: an unpinned
# rebuild on 2026-10-12 pulled 9.27, and ORT 1.23's cuDNN-frontend conv path
# failed every Conv with HEURISTIC_QUERY_FAILED (every request 503). Move this
# pin only together with ONNXRUNTIME_GPU_VERSION, after a GPU load test.
ARG ONNXRUNTIME_GPU_VERSION=""
ARG CUDNN_VERSION="9.8.0.87"
RUN if [ -n "$ONNXRUNTIME_GPU_VERSION" ]; then \
        pip uninstall -y onnxruntime \
        && pip install --no-cache-dir \
            "onnxruntime-gpu[cuda,cudnn]==${ONNXRUNTIME_GPU_VERSION}" \
            "nvidia-cudnn-cu12==${CUDNN_VERSION}" \
        && find /usr/local/lib/python3.12/site-packages/nvidia -type d -name lib \
            > /etc/ld.so.conf.d/nvidia-pip-wheels.conf \
        && ldconfig; \
    fi

COPY models/ /app/models/

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /home/appuser/.insightface \
    && chown -R appuser:appuser /app /home/appuser/.insightface

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8001/healthz || exit 1

CMD ["python", "-m", "uvicorn", "zepiris.ml_inference.app:app", "--host", "0.0.0.0", "--port", "8001"]
