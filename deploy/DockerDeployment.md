# VA-AI-CAC Docker/Podman Deployment Guide

## Overview

This guide documents the containerization of the VA-AI-CAC (Veterans Affairs AI-based Coronary Artery Calcium scoring) application for deployment with GPU acceleration using Docker/Podman.

## Requirements Deviation

### Original Specification (requirements.txt)
- CUDA 11.3
- PyTorch 1.12.1+cu113
- Torchvision 0.13.1+cu113

### Actual Implementation (Dockerfile.updated)
- CUDA 11.8
- PyTorch 2.0.1+cu118
- Torchvision 0.15.2

### Why We Deviated

The original requirements could not be containerized due to:

1. **PyTorch 1.12.1 wheels archived** - PyTorch removed old wheel distributions from their repository
2. **CUDA 11.3 base images unavailable** - NVIDIA doesn't provide CUDA 11.3 images for Ubuntu 22.04
3. **Ubuntu 20.04 PPA failures** - Deadsnakes PPA had connectivity/authentication issues
4. **CUDA version compatibility** - Initial builds used CUDA 13.0, incompatible with host driver (CUDA 12.0.2)
5. **MONAI dependency conflicts** - `monai[all]==1.4.0` automatically upgraded PyTorch to incompatible versions

### Working Solution

- **Base Image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Python**: 3.10 (native in Ubuntu 22.04, no PPA needed)
- **PyTorch**: 2.0.1+cu118
- **CUDA**: 11.8 (backward compatible with NVIDIA driver 535.129.03)
- **MONAI**: 1.4.0 (base package, without `[all]` to avoid dependency issues)

## Quick Start

### Build the Container

```bash
# Run from the repository root: the Dockerfile is in deploy/, but the build context is the root
cd /path/to/VA-AI-CAC

# Remove the previous container and image of this app only
podman rm -f va-ai-cac >/dev/null
podman rmi -f localhost/va-ai-cac >/dev/null

# Build from deploy/Dockerfile
podman build --no-cache -f deploy/Dockerfile -t va-ai-cac:latest .
```

### Run with GPU Support

```bash
# Settings: copy the template once and fill it in (deploy/.env is git-ignored and never baked into the image)
cp deploy/.env.example deploy/.env

# The container runs as uid 10001, so it must be able to write to the storage folder
mkdir -p storage && sudo chown 10001:10001 storage

podman run -d --name va-ai-cac \
  --device nvidia.com/gpu=all \
  --shm-size=2g \
  --env-file deploy/.env \
  -p 5000:25000 \
  -v $(pwd)/storage:/app/storage:Z \
  va-ai-cac:latest
```

- `--device nvidia.com/gpu=all` uses the NVIDIA Container Toolkit's CDI spec (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`).
- On SELinux hosts, allow GPU device access with `sudo setsebool -P container_use_devices=1` rather than `--security-opt=label=disable`, which turns off SELinux separation for the whole container. `:Z` relabels the storage folder for this container.
- The model weights are baked into the image and checked against `MODEL_SHA256` before loading, so no model volume is needed.

### Verify GPU Access

```bash
# Check container logs
podman logs va-ai-cac | grep "Device:"
# Expected: Device: cuda:0

# Verify PyTorch version
podman exec va-ai-cac python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU Available: {torch.cuda.is_available()}')"
# Expected: PyTorch: 2.0.1+cu118, CUDA: 11.8, GPU Available: True

# Test GPU
podman exec va-ai-cac nvidia-smi
```

## Running Predictions

### Health Check

```bash
curl http://localhost:5000/health
```

### Submit CT Study

```bash
curl -X POST http://localhost:5000/predict \
  -F "study_zip=@/path/to/CT_Study.zip" \
  -F "save_masks=true"
```

## Key Configuration Details

### Dockerfile Highlights

```dockerfile
# CUDA 11.8 base image (compatible with driver 535+)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Pin setuptools to avoid deprecation warnings
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel

# Install PyTorch with explicit CUDA 11.8 index
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchmetrics==1.5.2

# Install MONAI base (without [all] to prevent PyTorch upgrade)
RUN pip install --no-cache-dir "monai==1.4.0"

# Force reinstall PyTorch after MONAI dependencies
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1 \
    torchvision==0.15.2
```

### Host Requirements

- **NVIDIA Driver**: 535.129.03 or newer (CUDA 12.0.2+ support)
- **Podman**: 4.9+ or Docker 20.10+
- **NVIDIA Container Toolkit**: For GPU passthrough
- **OS**: Linux (tested on RHEL/CentOS)

## Performance

| Configuration | Processing Speed | Time (120-slice study) |
|---------------|------------------|------------------------|
| GPU (cuda:0) | 15-25 slices/sec | 5-8 seconds |
| CPU fallback | 2-3 slices/sec | 40-60 seconds |

## Troubleshooting

### Container Uses CPU Instead of GPU

```bash
# Verify GPU devices are accessible
podman exec va-ai-cac ls -la /dev/nvidia*

# Check CUDA availability
podman exec va-ai-cac python -c "import torch; print(torch.cuda.is_available())"

# If false, ensure devices are passed with --device flags
```

### Wrong PyTorch Version in Container

```bash
# Check version
podman exec va-ai-cac python -c "import torch; print(torch.__version__)"

# If incorrect, remove this app's container and image, then rebuild from the repository root
podman rm -f va-ai-cac
podman rmi -f localhost/va-ai-cac
podman build --no-cache --pull -f deploy/Dockerfile -t va-ai-cac:latest .
```

### Port 5000 Already in Use

```bash
# Find process using the port
sudo lsof -i :5000

# Use different host port
podman run -d --name va-ai-cac -p 5001:25000 ... va-ai-cac:latest
```

### CUDA Initialization Error

If you see "CUDA initialization failed" or "driver too old" errors:

1. **Check host driver version**:
   ```bash
   nvidia-smi
   ```
   Should show driver 535+ with CUDA 12.0+

2. **Verify container CUDA version**:
   ```bash
   podman exec va-ai-cac nvcc --version
   ```
   Should show CUDA 11.8

3. **Ensure driver compatibility**: CUDA 11.8 requires driver 450.80.02+ (you have 535.129.03 ✓)

## Version Compatibility Matrix

| Component | requirements.txt | Actual Container | Status |
|-----------|------------------|------------------|--------|
| Python | 3.10 | 3.10 | ✅ Match |
| PyTorch | 1.12.1+cu113 | 2.0.1+cu118 | ⚠️ Upgraded (wheels unavailable) |
| torchvision | 0.13.1+cu113 | 0.15.2 | ⚠️ Upgraded |
| MONAI | 1.4.0 | 1.4.0 | ✅ Match |
| numpy | 1.26.4 | 1.26.4 | ✅ Match |
| scipy | 1.13.1 | 1.13.1 | ✅ Match |
| SimpleITK | 2.4.1 | 2.4.1 | ✅ Match |
| pydicom | 2.4.4 | 2.4.5 | ⚠️ Upgraded (CVE-2026-32711) |
| Pillow | 11.1.0 | 12.3.0 | ⚠️ Upgraded (security fixes) |

torch 2.0.1 and MONAI 1.4.0 also have published CVEs. Upgrading them is planned in a separate change, because the scores must first be shown to be identical.

## Container Management

```bash
# View running containers
podman ps

# Stop container
podman stop va-ai-cac

# Start stopped container
podman start va-ai-cac

# Restart container
podman restart va-ai-cac

# View logs (live)
podman logs -f va-ai-cac

# Remove container
podman rm -f va-ai-cac

# Execute commands inside container
podman exec -it va-ai-cac bash
```

## Storage Cleanup

The API deletes each uploaded zip and its extracted DICOMs as soon as the request finishes, whether it succeeded or failed. Masks and debug CSVs are kept for `OUTPUT_RETENTION_HOURS` (default 24) and removed automatically after that. Debug CSVs are only written when `WRITE_DEBUG_CSVS=true`.

In S3 mode, add an S3 lifecycle rule on the `S3_PREFIX` prefix that expires objects after the same retention period.

```bash
# Check storage usage
du -sh storage/outputs/masks
```

## Security and Data Handling

- **Secrets:** keep settings in `deploy/.env` (git-ignored, passed with `--env-file`). Prefer an IAM role (EC2 instance profile, ECS task role) over static `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.
- **S3 bucket:** must stay private, with S3 Block Public Access on. `/masks/...` redirects to a presigned URL that expires after `S3_URL_EXPIRATION` seconds.
- **Uploads:** limited by `MAX_UPLOAD_MB` (request size), `MAX_UNCOMPRESSED_MB` (total extracted size) and `MAX_ZIP_FILES` (file count).
- **Errors:** the client gets a generic message and the `request_id`; the full traceback is only in the container log (`podman logs`).
- **Runtime:** gunicorn as the non-root user `appuser` (uid 10001). There is no debug mode in the container; `FLASK_DEBUG` only affects local `python app.py` runs, and then only on 127.0.0.1.
- **Model integrity:** the build and the app both check the weights against `MODEL_SHA256` before loading, and loading uses `weights_only=True`. The upstream release publishes no checksum; the pinned value was computed from the v1.0.0 asset and should be confirmed with the model author.
- **Build context:** `.dockerignore` allows only the API's source files into the build, so `.env`, `.git`, `storage/` and local model files never end up in the image.

## Running Multiple Instances

To run 2 instances of the container:

```bash
# Instance 1
podman run --rm -d \
  -p 5000:25000 \
  --shm-size=2g \
  --device nvidia.com/gpu=all \
  --env-file deploy/.env \
  -v ${PWD}/storage-1:/app/storage:Z \
  --name va-ai-cac-1 \
  va-ai-cac

# Instance 2
podman run --rm -d \
  -p 5001:25000 \
  --shm-size=2g \
  --device nvidia.com/gpu=all \
  --env-file deploy/.env \
  -v ${PWD}/storage-2:/app/storage:Z \
  --name va-ai-cac-2 \
  va-ai-cac
```

**Key points:**
- Different container names (`va-ai-cac-1`, `va-ai-cac-2`)
- Different host ports (`5000`, `5001`)
- Different storage volumes to avoid conflicts

## References

- **NVIDIA CUDA Compatibility**: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **PyTorch CUDA Builds**: https://pytorch.org/get-started/locally/
- **MONAI Documentation**: https://docs.monai.io/
- **Podman GPU Access**: https://github.com/containers/podman/blob/main/docs/tutorials/podman-with-gpus.md
- **VA-AI-CAC Paper**: https://doi.org/10.1056/AIoa2400937

## Files

- `Dockerfile` - Working Dockerfile with CUDA 11.8 + PyTorch 2.0.1
- `requirements.Dockerfile.txt` - Python dependencies matching container build
- `LOCAL_SETUP.md` - Local development setup (without containers)

## Author

**Muazzam Khan**  
Deployment Date: 2026-06-30

## Model Attribution

**VA-AI-CAC Model**  
Creator: Raffi Hagopian, MD  
U.S. Department of Veterans Affairs  
Citation: Hagopian, R., et al. (2024). NEJM AI. DOI: 10.1056/AIoa2400937
