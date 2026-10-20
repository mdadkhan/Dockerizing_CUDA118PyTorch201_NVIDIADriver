# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# VA-AI-CAC – Dockerfile
# 
# Container CUDA: 11.8.0 + cuDNN 8
# Python: 3.10 (Ubuntu 22.04)
# PyTorch: 2.0.1 (compiled for CUDA 11.8)
# Flask API: port 25000 (served by gunicorn, as a non-root user)
#
# Compatible with NVIDIA driver 535+ (supports CUDA 12.x via backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    libopenslide-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Virtual environment ───────────────────────────────────────────────────────
ENV VENV_PATH=/app/.venv
RUN python -m venv "$VENV_PATH"
ENV PATH="$VENV_PATH/bin:$PATH"

# ── Python dependencies ───────────────────────────────────────────────────────
# Pin setuptools<81 to avoid pkg_resources deprecation warning
# Cache buster to force rebuild: 2026-07-02-v3
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    # PyTorch 2.0.1 for CUDA 11.8 (CRITICAL - must match driver compatibility)
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1 \
        torchvision==0.15.2 \
        torchmetrics==1.5.2 && \
    # Core numerical libraries, medical imaging, MONAI, and web framework in one layer
    pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        scikit-learn==1.3.2 \
        pandas==2.0.3 \
        matplotlib==3.7.2 \
        ipywidgets==7.8.1 \
        "Pillow==12.3.0" \
        itk==5.3.0 \
        SimpleITK==2.4.1 \
        nibabel==5.3.2 \
        "pydicom==2.4.5" \
        python-gdcm==3.0.24.1 \
        opencv-python==4.8.1.78 \
        "monai==1.4.0" \
        "einops==0.8.0" \
        psutil \
        tqdm \
        "Flask==3.0.0" \
        "werkzeug==3.0.0" \
        "boto3==1.35.0" \
        "gunicorn==23.0.0" && \
    # CRITICAL: Force reinstall correct PyTorch version (MONAI may have upgraded it)
    pip install --no-cache-dir --force-reinstall --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1 \
        torchvision==0.15.2

# ── Application code ───────────────────────────────────────────────────────────
# Only the API's own modules; .dockerignore keeps everything else out of the build context.
COPY app.py cac_inference_service.py s3_storage.py \
     filter_series.py dataset_generator_inference.py processing.py visualization.py \
     ./

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p \
    storage/incoming \
    storage/outputs/masks \
    storage/outputs/debug \
    model

# ── Model weights ─────────────────────────────────────────────────────────────
# torch.load runs pickled code, so the download must match this SHA-256 or the build fails.
# The app checks the same value again (MODEL_SHA256) before loading.
ARG MODEL_SHA256=cfded8253e64b81b963a314cb5adc90d30a1f21574638366b80c783676c1733d
RUN MODEL_URL="https://github.com/Raffi-Hagopian/AI-CAC/releases/download/v1.0.0/va_non_gated_ai_cac_model.pth" && \
    echo "Downloading model weights from GitHub..." && \
    wget -q --show-progress -O /app/model/va_non_gated_ai_cac_model.pth "$MODEL_URL" && \
    echo "${MODEL_SHA256}  /app/model/va_non_gated_ai_cac_model.pth" | sha256sum -c - && \
    echo "Model downloaded and verified: $(du -sh /app/model/va_non_gated_ai_cac_model.pth)"

# ── Non-root user ─────────────────────────────────────────────────────────────
# Only storage/ is writable; the code and the model stay owned by root.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app/storage
USER appuser

# ── Environment ───────────────────────────────────────────────────────────────
ENV MODEL_CHECKPOINT_FILE=model/va_non_gated_ai_cac_model.pth \
    MODEL_SHA256=${MODEL_SHA256} \
    MPLBACKEND=Agg

EXPOSE 25000

# One worker: each worker loads its own copy of the model onto the GPU. Inference on a
# study can take minutes, hence the long timeout. Override with GUNICORN_CMD_ARGS.
CMD ["gunicorn", "--bind", "0.0.0.0:25000", "--workers", "1", "--threads", "4", "--timeout", "900", "app:app"]