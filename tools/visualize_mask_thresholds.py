import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    image = Image.open(mask_path).convert("L")
    mask = np.asarray(image, dtype=np.float32) / 255.0
    return mask


def _find_first_image(folder: Path) -> Path:
    if not folder.is_dir():
        raise FileNotFoundError(f"Mask folder not found: {folder}")
    candidates: List[Path] = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}],
        key=lambda x: x.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No image files found under {folder}")
    return candidates[0]


def _apply_threshold(mask: np.ndarray, threshold: float) -> np.ndarray:
    binary = (mask >= threshold).astype(np.float32)
    return binary


def visualize_masks(base_dir: Path, thresholds: Iterable[float], output_path: Path | None, show: bool) -> None:
    base_dir = base_dir.expanduser().resolve()
    mask_sources: List[Tuple[str, Path]] = [
        ("mask", base_dir / "mask"),
        ("sa3d_mask", base_dir / "sa3d_mask"),
        ("splat_mask", base_dir / "splat_mask"),
    ]

    entries: List[Tuple[str, Path, np.ndarray]] = []
    for label, folder in mask_sources:
        try:
            image_path = _find_first_image(folder)
            mask = _load_mask(image_path)
            entries.append((label, image_path, mask))
        except FileNotFoundError as exc:
            print(f"[Skip] {exc}", file=sys.stderr)

    if not entries:
        raise RuntimeError(f"No mask images found in {base_dir}")

    thresholds = list(thresholds)
    if not thresholds:
        raise ValueError("At least one threshold must be provided")

    n_rows = len(entries)
    n_cols = len(thresholds) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for row_idx, (label, image_path, mask) in enumerate(entries):
        axes[row_idx][0].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_idx][0].set_title(f"{label}: raw\n{image_path.name}")
        axes[row_idx][0].axis("off")

        for col_idx, threshold in enumerate(thresholds, start=1):
            binary = _apply_threshold(mask, threshold)
            axes[row_idx][col_idx].imshow(binary, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_idx][col_idx].set_title(f"{label}: threshold {threshold:.2f}")
            axes[row_idx][col_idx].axis("off")

    fig.suptitle(f"Mask threshold comparison @ {base_dir}")
    fig.tight_layout()

    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize mask images under different thresholds.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(r"../output/6189be9c-splat/train/ours_94_0_1"),
        help="Directory containing mask/, sa3d_mask/, and splat_mask/ subfolders.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="List of MASK_THRESHOLD values to apply.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the comparison figure (e.g., output.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the matplotlib window in addition to saving the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize_masks(args.base_dir, args.thresholds, args.output, args.show)


if __name__ == "__main__":
    main()
