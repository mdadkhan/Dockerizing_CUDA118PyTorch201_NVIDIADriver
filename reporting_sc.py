import math
import re
import tempfile
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import (
    generate_uid,
    SecondaryCaptureImageStorage,
    ExplicitVRLittleEndian,
    PYDICOM_IMPLEMENTATION_UID,
)


def find_source_study_folder(studies_root: Path, study_name: str) -> Path | None:
    candidate = studies_root / study_name / study_name
    if candidate.exists() and candidate.is_dir():
        return candidate

    candidate2 = studies_root / study_name
    if candidate2.exists() and candidate2.is_dir():
        return candidate2

    return None


def get_first_dicom_file(folder: Path) -> Path | None:
    for ext in ("*.dcm", "*.dicom", "*"):
        files = sorted(folder.glob(ext))
        for f in files:
            if f.is_file():
                try:
                    _ = dcmread(str(f), stop_before_pixels=True, force=True)
                    return f
                except Exception:
                    pass
    return None


def get_slice_position(ds):
    if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
        return None

    try:
        ipp = np.array([float(x) for x in ds.ImagePositionPatient], dtype=float)
        iop = [float(x) for x in ds.ImageOrientationPatient]

        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:], dtype=float)
        normal = np.cross(row, col)

        return float(np.dot(ipp, normal))
    except Exception:
        return None


# Shown on every report page and written into every Secondary Capture. The DICOM text
# uses a plain hyphen because the copied patient tags keep the source's character set.
RESEARCH_USE_LABEL = "Research use only \u2013 not for diagnostic use"
RESEARCH_USE_LABEL_DICOM = "Research use only - not for diagnostic use"


def extract_png_slice_number(png_path: Path) -> int | None:
    """
    The slice index the model saved the PNG under: the 0-based position of the slice in
    the volume (see visualization.save_vol_masks).

    Examples:
      28.png -> 28
      28_no_mask.png -> 28
      0030.png -> 30
      0030_overlay.png -> 30
    """
    m = re.match(r"(\d+)", png_path.stem)
    return int(m.group(1)) if m else None


def read_axial_position(ds) -> float | None:
    """The z value of ImagePositionPatient, the key inference sorts the slices by."""
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        return None


def sorted_slices_from_table(slice_table: Path, study_name: str) -> list[tuple[Path, float]]:
    """
    The study's slices in volume order, from the table inference writes
    (dicom_input_one_series.csv). That table holds exactly the series the model used.
    """
    df = pd.read_csv(slice_table, dtype=str)
    rows = df[df["StudyName"] == study_name]
    slices = [(Path(p), float(z)) for p, z in zip(rows["DICOMFilePath"], rows["AxialPosition"])]
    missing = [p for p, _ in slices if not p.is_file()]
    if missing:
        raise ValueError(f"{len(missing)} DICOM file(s) listed in {slice_table} were not found, e.g. {missing[0]}")
    # Stable sort by axial position, as processing.load_pydicom_slices_by_axial_cord does.
    slices.sort(key=lambda s: s[1])
    return slices


def sorted_slices_from_folder(study_dicom_folder: Path) -> list[tuple[Path, float]]:
    """
    The study's slices in volume order, read from the DICOM folder. Only valid when the
    folder holds one series, because inference picks one series per study and the folder
    alone cannot say which.
    """
    slices = []
    series_uids = set()
    for f in sorted(study_dicom_folder.glob("*")):
        if not f.is_file():
            continue
        try:
            ds = dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        z = read_axial_position(ds)
        if z is None:
            continue
        series_uids.add(str(getattr(ds, "SeriesInstanceUID", "")))
        slices.append((f, z))

    if len(series_uids) > 1:
        raise ValueError(
            f"{study_dicom_folder} holds {len(series_uids)} series, so the slices the model used "
            "cannot be told apart; pass --slice_table (results/dicom_input_one_series.csv from inference)"
        )
    slices.sort(key=lambda s: s[1])
    return slices


def sort_pngs_within_slice(png_list: list[Path]) -> list[Path]:
    """
    Keep a predictable per-slice display order.
    Preferred:
      original PNG first
      *_no_mask.png second
      everything else after
    """
    def key_func(p: Path):
        stem_lower = p.stem.lower()
        if stem_lower == re.match(r"(\d+)", p.stem).group(1):
            return (0, p.name.lower())
        if stem_lower.endswith("_no_mask"):
            return (1, p.name.lower())
        return (2, p.name.lower())

    return sorted(png_list, key=key_func)


def group_pngs_by_slice(study_png_folder: Path):
    pngs = sorted(study_png_folder.glob("*.png"))
    grouped = {}
    unmatched = []

    for png in pngs:
        slice_num = extract_png_slice_number(png)
        if slice_num is None:
            unmatched.append(png)
            continue
        grouped.setdefault(slice_num, []).append(png)

    for slice_num in grouped:
        grouped[slice_num] = sort_pngs_within_slice(grouped[slice_num])

    return grouped, unmatched

def match_pngs_to_slices(study_png_folder: Path, slices: list[tuple[Path, float]]):
    """
    Match each mask PNG to its DICOM by position in the sorted volume.

    PNG N is slice index N of the volume the model saw, i.e. the N-th DICOM (0-based)
    when the series is sorted by ascending axial position. The DICOM file name plays
    no part. Pages are ordered superior to inferior (highest axial position first).

    PNGs whose number is not a slice of the volume are returned as unmatched.
    """
    grouped_pngs, unmatched_pngs = group_pngs_by_slice(study_png_folder)

    matched_groups = []
    unmatched_slice_groups = []

    for slice_index, png_group in grouped_pngs.items():
        if slice_index >= len(slices):
            unmatched_slice_groups.extend(png_group)
            continue
        dcm_path, slice_pos = slices[slice_index]
        matched_groups.append({
            "slice_index": slice_index,
            "png_group": png_group,
            "dcm_path": dcm_path,
            "slice_pos": slice_pos,
        })

    ordered_groups = sorted(matched_groups, key=lambda x: (x["slice_pos"], x["slice_index"]), reverse=True)

    ordered_pngs = []
    matched_dicoms = []

    for group in ordered_groups:
        ordered_pngs.extend(group["png_group"])
        matched_dicoms.append(group["dcm_path"])

    unmatched_all = unmatched_pngs + unmatched_slice_groups
    return ordered_pngs, matched_dicoms, unmatched_all


def print_order_check(study_name: str, matched_dicoms: list[Path]):
    vals = []

    for dcm_path in matched_dicoms:
        try:
            ds = dcmread(str(dcm_path), stop_before_pixels=True, force=True)
            pos = get_slice_position(ds)
            if pos is not None:
                vals.append(pos)
        except Exception:
            pass

    if len(vals) >= 2:
        increasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        decreasing = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        print(
            f"[ORDER CHECK] Study={study_name} | matched_positions={len(vals)} | "
            f"increasing={increasing} | decreasing={decreasing} | "
            f"first={vals[0]:.4f} | last={vals[-1]:.4f}"
        )
    else:
        print(f"[ORDER CHECK] Study={study_name} | not enough matched geometry data to verify order")


def make_sc_from_image(
    image_path: Path,
    source_ds,
    out_path: Path,
    series_uid: str,
    instance_number: int,
    series_number: int = 900,
    series_description: str = "VA-AI-CAC Report (research use only)",
):
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rows, cols, _ = img_rgb.shape

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

    ds = Dataset()
    ds.file_meta = file_meta

    for tag_name in [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "StudyInstanceUID",
        "StudyID",
        "AccessionNumber",
        "StudyDate",
        "StudyTime",
        "ReferringPhysicianName",
        "StudyDescription",
        "InstitutionName",
    ]:
        if hasattr(source_ds, tag_name):
            setattr(ds, tag_name, getattr(source_ds, tag_name))

    now = datetime.now()
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "SC"

    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.InstanceNumber = instance_number
    ds.SeriesDescription = series_description
    ds.ImageComments = RESEARCH_USE_LABEL_DICOM
    ds.BurnedInAnnotation = "YES"

    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.SeriesTime = now.strftime("%H%M%S")

    ds.ConversionType = "WSD"
    ds.ImageType = r"DERIVED\SECONDARY"

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = img_rgb.tobytes()

    ds.PatientOrientation = ""
    ds.SecondaryCaptureDeviceManufacturer = "VA-AI"
    ds.SecondaryCaptureDeviceManufacturerModelName = "VA-AI-CAC Report Generator"
    ds.SecondaryCaptureDeviceSoftwareVersions = "1.0"

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dcmwrite(str(out_path), ds, write_like_original=False)


def generate_pdf_and_sc_reports(mask_root, score_file, pdf_output_dir, studies_root, sc_output_dir, slice_table=None):
    mask_root = Path(mask_root)
    pdf_output_dir = Path(pdf_output_dir)
    studies_root = Path(studies_root)
    sc_output_dir = Path(sc_output_dir)
    slice_table = Path(slice_table) if slice_table else None

    if slice_table is not None and not slice_table.is_file():
        print(f"Error: Slice table {slice_table} does not exist.")
        return

    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    sc_output_dir.mkdir(parents=True, exist_ok=True)

    if not mask_root.exists():
        print(f"Error: Input directory {mask_root} does not exist.")
        return

    scores_dict = {}
    if Path(score_file).exists():
        try:
            df = pd.read_csv(score_file)
            scores_dict = dict(zip(df["StudyName"].astype(str), df["AI-CAC"]))
        except Exception as e:
            print(f"Warning: Could not read score file {score_file}: {e}")

    studies = [d for d in mask_root.iterdir() if d.is_dir()]
    if not studies:
        print(f"No study subfolders found in {mask_root}")
        return

    for study_path in studies:
        study_name = study_path.name
        study_score = scores_dict.get(study_name, "N/A")

        source_study_folder = find_source_study_folder(studies_root, study_name)
        if source_study_folder is None:
            print(f"Skipping {study_name}: No matching source study folder found in {studies_root}")
            continue

        source_dcm_path = get_first_dicom_file(source_study_folder)
        if source_dcm_path is None:
            print(f"Skipping {study_name}: No readable DICOM found in {source_study_folder}")
            continue

        source_ds = dcmread(str(source_dcm_path), force=True)

        try:
            if slice_table is not None:
                slices = sorted_slices_from_table(slice_table, study_name)
            else:
                slices = sorted_slices_from_folder(source_study_folder)
        except ValueError as e:
            print(f"Skipping {study_name}: {e}")
            continue

        if not slices:
            print(f"Skipping {study_name}: No slices found for this study in {slice_table or source_study_folder}")
            continue

        images, matched_dicoms, unmatched_pngs = match_pngs_to_slices(
            study_png_folder=study_path,
            slices=slices,
        )

        print_order_check(study_name, matched_dicoms)

        if unmatched_pngs:
            print(f"[WARN] {study_name}: {len(unmatched_pngs)} PNG(s) did not match a slice of the sorted volume ({len(slices)} slices)")
            for p in unmatched_pngs[:10]:
                print(f"        Unmatched PNG: {p.name}")

        if not images:
            print(f"Skipping {study_name}: No matched PNG files found after DICOM ordering.")
            continue

        safe_study_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in study_name)
        study_pdf = pdf_output_dir / f"{safe_study_name}_AI-CAC_Report.pdf"

        study_sc_dir = sc_output_dir / safe_study_name
        study_sc_dir.mkdir(parents=True, exist_ok=True)

        cols = 2
        rows = 4
        imgs_per_page = cols * rows
        num_pages = math.ceil(len(images) / imgs_per_page)

        sc_series_uid = generate_uid()

        with PdfPages(study_pdf) as pdf:
            for p in range(num_pages):
                fig, axes = plt.subplots(rows, cols, figsize=(16, 32))
                fig.suptitle(
                    f"Study Name: {study_name}\n\nVA-AI-CAC: {study_score}",
                    fontsize=30,
                    fontweight="bold",
                    y=0.97,
                )

                fig.text(
                    0.5,
                    0.995,
                    RESEARCH_USE_LABEL,
                    ha="center",
                    va="top",
                    fontsize=22,
                    fontweight="bold",
                    color="red",
                )

                fig.text(
                    0.5,
                    0.02,
                    f"Page {p + 1} of {num_pages}",
                    ha="center",
                    fontsize=20,
                    fontweight="bold",
                )

                ax_flat = axes.flatten()
                start_idx = p * imgs_per_page

                for i in range(imgs_per_page):
                    img_idx = start_idx + i
                    ax = ax_flat[i]

                    if img_idx < len(images):
                        img_path = images[img_idx]
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            ax.imshow(img, interpolation="nearest")
                            ax.set_anchor("N")
                            ax.set_title(img_path.name, fontsize=20, y=0.92, pad=0)
                        else:
                            ax.text(0.5, 0.5, "Load Error", ha="center", va="center")

                    ax.axis("off")

                plt.subplots_adjust(
                    left=0.03,
                    right=0.97,
                    top=0.90,
                    bottom=0.03,
                    hspace=0.12,
                    wspace=0.05,
                )

                pdf.savefig(fig)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_png = Path(tmp.name)

                fig.savefig(temp_png, dpi=150, bbox_inches="tight")
                plt.close(fig)

                sc_path = study_sc_dir / f"{safe_study_name}_SC_Page_{p + 1:03d}.dcm"
                make_sc_from_image(
                    image_path=temp_png,
                    source_ds=source_ds,
                    out_path=sc_path,
                    series_uid=sc_series_uid,
                    instance_number=p + 1,
                    series_number=900,
                    series_description="VA-AI-CAC Report (research use only)",
                )

                temp_png.unlink(missing_ok=True)

        print(f"Generated PDF: {study_pdf}")
        print(f"Generated SC DICOMs in: {study_sc_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate one PDF and SC DICOM set per study.")
    parser.add_argument("--input_dir", type=str, default="results2/predicted_masks")
    parser.add_argument("--score_file", type=str, default="results2/scores.csv")
    parser.add_argument("--pdf_output_dir", type=str, default="results2/reports_pdf")
    parser.add_argument("--studies_dir", type=str, default="studies2")
    parser.add_argument("--sc_output_dir", type=str, default="results2/reports_sc")
    parser.add_argument(
        "--slice_table",
        type=str,
        default=None,
        help="dicom_input_one_series.csv written by inference. Required when a study folder holds more than one series.",
    )

    args = parser.parse_args()

    generate_pdf_and_sc_reports(
        mask_root=args.input_dir,
        score_file=args.score_file,
        pdf_output_dir=args.pdf_output_dir,
        studies_root=args.studies_dir,
        sc_output_dir=args.sc_output_dir,
        slice_table=args.slice_table,
    )