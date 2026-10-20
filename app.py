print("app.py was executed")

import os

# Force Matplotlib to use a non-GUI backend.
# Needed because Flask request handlers may run outside the main thread.
os.environ["MPLBACKEND"] = "Agg"

import uuid
import time
import shutil
import zipfile
from pathlib import Path as FilePath

from flask import Flask, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename

from cac_inference_service import CACSInferenceService, StudyInputError, DEFAULT_MODEL_SHA256
from s3_storage import S3StorageService


def env_flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"true", "1", "yes"}


# -----------------------------
# Configuration
# -----------------------------

MODEL_CHECKPOINT_FILE = os.environ.get(
    "MODEL_CHECKPOINT_FILE",
    os.path.join("model", "va_non_gated_ai_cac_model.pth"),
)
MODEL_SHA256 = os.environ.get("MODEL_SHA256", DEFAULT_MODEL_SHA256)

# S3 Configuration
USE_S3_STORAGE = env_flag("USE_S3_STORAGE")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "RADFLOW/outputs/ai-cac-outputs")
S3_URL_EXPIRATION = int(os.environ.get("S3_URL_EXPIRATION", "3600"))  # 1 hour default

# Upload limits (zip-bomb protection)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
MAX_UNCOMPRESSED_MB = int(os.environ.get("MAX_UNCOMPRESSED_MB", "4096"))
MAX_ZIP_FILES = int(os.environ.get("MAX_ZIP_FILES", "10000"))

# PHI retention: inputs are deleted after every request; outputs are kept this long (0 = no limit)
OUTPUT_RETENTION_HOURS = float(os.environ.get("OUTPUT_RETENTION_HOURS", "24"))
WRITE_DEBUG_CSVS = env_flag("WRITE_DEBUG_CSVS")
# Input folders left behind by a killed worker are removed after this many hours.
INCOMING_STALE_HOURS = 6

BASE_STORAGE = FilePath("storage")

# Incoming DICOM uploads only (always local for processing)
INCOMING_ROOT = BASE_STORAGE / "incoming"

# Outputs - can be local or S3
OUTPUT_ROOT = BASE_STORAGE / "outputs"
MASK_ROOT = OUTPUT_ROOT / "masks"
DEBUG_ROOT = OUTPUT_ROOT / "debug"

# Always create incoming directory (needed for DICOM extraction)
INCOMING_ROOT.mkdir(parents=True, exist_ok=True)

# Create output directories only if not using S3
if not USE_S3_STORAGE:
    MASK_ROOT.mkdir(parents=True, exist_ok=True)
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Flask app
# -----------------------------

app = Flask(__name__)
# Requests larger than this are rejected with 413 before the upload is stored.
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# -----------------------------
# Initialize S3 Storage (if enabled)
# -----------------------------

s3_service = None

if USE_S3_STORAGE:
    if not S3_BUCKET_NAME:
        raise ValueError(
            "USE_S3_STORAGE is enabled but S3_BUCKET_NAME is not set. "
            "Set S3_BUCKET_NAME environment variable."
        )
    
    print("Initializing S3 storage...")
    print(f"  Bucket: {S3_BUCKET_NAME}")
    print(f"  Prefix: {S3_PREFIX}")
    print(f"  URL Expiration: {S3_URL_EXPIRATION}s")
    
    s3_service = S3StorageService(
        bucket_name=S3_BUCKET_NAME,
        prefix=S3_PREFIX,
        presigned_url_expiration=S3_URL_EXPIRATION,
    )
    
    print("S3 storage initialized successfully.")
else:
    print("Using local file storage (S3 disabled).")


# -----------------------------
# Auto-load AI-CAC model at startup
# -----------------------------

print("Loading AI-CAC service at startup...")
print("Model checkpoint:", MODEL_CHECKPOINT_FILE)
print("Mask root:", MASK_ROOT)
print("Debug root:", DEBUG_ROOT)

cac_service = CACSInferenceService(
    model_checkpoint_file=MODEL_CHECKPOINT_FILE,
    mask_root=str(MASK_ROOT),
    debug_root=str(DEBUG_ROOT),
    batch_size=int(os.environ.get("INFERENCE_BATCH_SIZE", "16")),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
    s3_service=s3_service,
    write_debug_csvs=WRITE_DEBUG_CSVS,
    model_sha256=MODEL_SHA256,
)

print("AI-CAC service loaded successfully.")


# -----------------------------
# Helpers
# -----------------------------

class UploadRejected(ValueError):
    """The uploaded zip breaks a safety limit. The message is safe to return to the client."""


def safe_extract_zip(zip_path: FilePath, extract_dir: FilePath):
    """
    Extract a zip file, rejecting path traversal and enforcing the file-count and
    total-uncompressed-size limits. Sizes are counted while extracting, not taken
    from the zip headers, which an attacker controls.
    """
    extract_dir = extract_dir.resolve()
    max_bytes = MAX_UNCOMPRESSED_MB * 1024 * 1024

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = [m for m in zip_ref.infolist() if not m.is_dir()]

        if len(members) > MAX_ZIP_FILES:
            raise UploadRejected(f"The zip contains more than {MAX_ZIP_FILES} files.")
        if sum(m.file_size for m in members) > max_bytes:
            raise UploadRejected(f"The zip expands to more than {MAX_UNCOMPRESSED_MB} MB.")

        written = 0
        for member in members:
            target_path = (extract_dir / member.filename).resolve()
            if not target_path.is_relative_to(extract_dir):
                raise UploadRejected("The zip contains an unsafe file path.")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(member) as src, open(target_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    written += len(chunk)
                    if written > max_bytes:
                        raise UploadRejected(f"The zip expands to more than {MAX_UNCOMPRESSED_MB} MB.")
                    dst.write(chunk)


def purge_old_request_dirs(root: FilePath, max_age_hours: float) -> None:
    """Delete per-request folders under root older than max_age_hours (0 = keep)."""
    if max_age_hours <= 0 or not root.is_dir():
        return
    cutoff = time.time() - max_age_hours * 3600
    for entry in root.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


def purge_expired_data() -> None:
    purge_old_request_dirs(INCOMING_ROOT, INCOMING_STALE_HOURS)
    purge_old_request_dirs(MASK_ROOT, OUTPUT_RETENTION_HOURS)
    purge_old_request_dirs(DEBUG_ROOT, OUTPUT_RETENTION_HOURS)


purge_expired_data()


def list_extracted_files(folder: FilePath, max_files: int = 20):
    """
    Lightweight debug helper.
    """
    files = []

    for path in folder.rglob("*"):
        if path.is_file():
            files.append(str(path))

        if len(files) >= max_files:
            break

    return files


def local_mask_path_to_url(mask_file: str) -> str:
    """
    Convert a stored mask (local path, or S3 key in S3 mode) into this API's /masks/... URL.
    In S3 mode that URL redirects to a short-lived presigned URL, so the bucket stays private.
    """
    s3_mask_prefix = f"{S3_PREFIX}/masks/".lstrip("/")
    if USE_S3_STORAGE and s3_service and mask_file.startswith(s3_mask_prefix):
        return f"/masks/{mask_file[len(s3_mask_prefix):]}"

    mask_path = FilePath(mask_file).resolve()
    relative_path = mask_path.relative_to(MASK_ROOT.resolve())
    return f"/masks/{relative_path.as_posix()}"


# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "success": True,
            "message": "AI-CAC Flask server is running.",
            "health_url": "/health",
            "predict_url": "/predict",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "model_loaded": True,
            "device": str(cac_service.device),
            "storage_mode": "s3" if USE_S3_STORAGE else "local",
        }
    )


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(
        {
            "success": False,
            "error": f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.",
        }
    ), 413


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a zipped DICOM study folder.

    Expected form-data:
      study_zip: file
      save_masks: optional true/false
    """
    if "study_zip" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "Missing file field 'study_zip'. Upload a zipped DICOM study.",
            }
        ), 400

    uploaded_file = request.files["study_zip"]

    if uploaded_file.filename is None or uploaded_file.filename.strip() == "":
        return jsonify(
            {
                "success": False,
                "error": "Empty filename.",
            }
        ), 400

    request_id = uuid.uuid4().hex
    purge_expired_data()

    save_masks_raw = request.form.get("save_masks", "true").strip().lower()
    save_masks = save_masks_raw in {"true", "1", "yes", "y"}

    filename = secure_filename(uploaded_file.filename)

    request_dir = INCOMING_ROOT / request_id
    zip_path = request_dir / filename

    # This is the only folder that should be passed to predict_folder().
    dicom_extract_dir = request_dir / "dicoms"

    request_dir.mkdir(parents=True, exist_ok=True)
    dicom_extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_file.save(zip_path)

        safe_extract_zip(zip_path, dicom_extract_dir)

        extracted_preview = list_extracted_files(dicom_extract_dir)

        print("REQUEST ID:", request_id)
        print("ZIP PATH:", zip_path)
        print("DICOM EXTRACT DIR:", dicom_extract_dir)
        print("MASK ROOT:", MASK_ROOT)
        print("DEBUG ROOT:", DEBUG_ROOT)
        print("EXTRACTED FILE PREVIEW:", extracted_preview)

        result = cac_service.predict_folder(
            dicom_root_dir=str(dicom_extract_dir),
            request_id=request_id,
            save_masks=save_masks,
        )

        for study_result in result.get("results", []):
            mask_urls = []

            # mask_files/mask_dir are server paths or S3 keys; only API URLs go to the client.
            for mask_file in study_result.pop("mask_files", []):
                try:
                    mask_urls.append(local_mask_path_to_url(mask_file))
                except Exception:
                    pass

            study_result.pop("mask_dir", None)
            study_result["mask_urls"] = mask_urls

        return jsonify(
            {
                "success": True,
                "request_id": request_id,
                **result,
            }
        ), 200

    except zipfile.BadZipFile:
        return jsonify(
            {
                "success": False,
                "request_id": request_id,
                "error": "Uploaded file is not a valid zip file.",
            }
        ), 400

    except UploadRejected as e:
        return jsonify(
            {
                "success": False,
                "request_id": request_id,
                "error": str(e),
            }
        ), 400

    except StudyInputError as e:
        return jsonify(
            {
                "success": False,
                "request_id": request_id,
                "error": str(e),
            }
        ), 422

    except Exception:
        # Details stay in the server log; the client only gets the request_id to quote.
        app.logger.exception("Prediction failed for request_id=%s", request_id)

        return jsonify(
            {
                "success": False,
                "request_id": request_id,
                "error": "Internal error while processing the study. Quote the request_id when reporting it.",
            }
        ), 500

    finally:
        # The uploaded zip and extracted DICOMs are PHI; never keep them after the request.
        shutil.rmtree(request_dir, ignore_errors=True)


@app.route("/masks/<path:filename>", methods=["GET"])
def get_mask(filename):
    """
    Serves saved mask PNGs from storage/outputs/masks (local mode), or redirects to a
    short-lived presigned S3 URL (S3 mode). The bucket itself must stay private.
    """
    if USE_S3_STORAGE and s3_service and not (MASK_ROOT / filename).is_file():
        s3_key = f"{S3_PREFIX}/masks/{filename}".lstrip("/")
        return redirect(s3_service.generate_presigned_url(s3_key))
    return send_from_directory(MASK_ROOT.resolve(), filename)


print("Reached bottom of app.py")
print("__name__ is:", __name__)

if __name__ == "__main__":
    # Local development only; the container runs gunicorn and never reaches this.
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "25000"))

    debug_mode = env_flag("FLASK_DEBUG")

    if debug_mode:
        # The Werkzeug debugger allows remote code execution, so never expose it beyond this machine.
        host = "127.0.0.1"
        print("⚠️  DEBUG MODE ENABLED - Auto-reload on code changes (listening on 127.0.0.1 only)")

    print("Starting AI-CAC Flask server...")
    print(f"Open: http://127.0.0.1:{port}/health")

    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        use_reloader=debug_mode,
    )