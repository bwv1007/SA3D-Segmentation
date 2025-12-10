"""Generate thresholded mask comparisons for a rendered scene (first view)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {folder}")
    return files


def _select_image(folder: Path, index: int) -> Path:
    files = _list_images(folder)
    index = max(0, min(index, len(files) - 1))
    return files[index]


def _load_mask(path: Path) -> np.ndarray:
    import cv2

    data = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    data = data.astype(np.float32)
    if data.max() > 1.5:
        data /= 255.0
    return data


def _save_mask(data: np.ndarray, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(data, 0.0, 1.0)
    cv2.imwrite(str(path), (array * 255.0).astype(np.uint8))


def _threshold_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    return (mask >= threshold).astype(np.float32)


def _format_threshold(value: float) -> str:
    return str(value).replace(".", "_")


def run_threshold_sweep(
    base_dir: Path,
    thresholds: Sequence[float],
    view_index: int,
    gt_dir: Optional[Path],
    output_dir: Optional[Path],
) -> None:
    base_dir = base_dir.resolve()
    sa3d_dir = base_dir / "sa3d_mask"
    splat_dir = base_dir / "splat_mask"
    combined_dir = base_dir / "mask"

    sa3d_path = _select_image(sa3d_dir, view_index)
    splat_path = _select_image(splat_dir, view_index)
    combined_path = _select_image(combined_dir, view_index)

    sa3d_mask = _load_mask(sa3d_path)
    splat_mask = _load_mask(splat_path)
    combined_original = _load_mask(combined_path)

    gt_mask: Optional[np.ndarray] = None
    if gt_dir is not None:
        gt_path = _select_image(gt_dir, view_index)
        gt_mask = _load_mask(gt_path)

    output_root = output_dir.resolve() if output_dir else base_dir / "mask_threshold_sweep"

    for threshold in thresholds:
        thr_label = _format_threshold(threshold)
        sweep_dir = output_root / f"thr_{thr_label}"
        thresholded_sa3d = _threshold_mask(sa3d_mask, threshold)
        combined_mask = np.clip(thresholded_sa3d * splat_mask, 0.0, 1.0)

        if gt_mask is not None:
            _save_mask(gt_mask, sweep_dir / "gt_mask.png")

        _save_mask(sa3d_mask, sweep_dir / "sa3d_mask_raw.png")
        _save_mask(thresholded_sa3d, sweep_dir / "sa3d_mask_thresh.png")
        _save_mask(splat_mask, sweep_dir / "splat_mask.png")
        _save_mask(combined_original, sweep_dir / "combined_original.png")
        _save_mask(combined_mask, sweep_dir / "combined_mask.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep mask thresholds for the first view.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Directory containing mask/, sa3d_mask/, splat_mask/ subfolders (e.g. .../train/ours_*).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Optional directory containing ground-truth masks to copy for comparison.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Threshold values to apply to the SA3D mask before combining with the splat mask.",
    )
    parser.add_argument(
        "--view-index",
        type=int,
        default=0,
        help="Index of the view to sample (default: 0, corresponds to the first sorted mask).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <base-dir>/mask_threshold_sweep when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_threshold_sweep(
        base_dir=args.base_dir,
        thresholds=args.thresholds,
        view_index=args.view_index,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
