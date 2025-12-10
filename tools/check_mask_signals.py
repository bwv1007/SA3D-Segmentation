import argparse
import os
import sys
from typing import Any, Optional

import torch

def load_tensor(path: str, key: Optional[str] = None) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data: Any = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        return data

    if isinstance(data, dict):
        if key is None:
            raise TypeError(
                "Loaded dict requires a key to extract tensor. Available keys: "
                + ", ".join(sorted(str(k) for k in data.keys()))
            )
        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in saved dict. Available keys: "
                + ", ".join(sorted(str(k) for k in data.keys()))
            )
        value = data[key]
        if isinstance(value, torch.Tensor):
            return value.cpu()
        raise TypeError(
            f"Value for key '{key}' is not a tensor (type: {type(value)})."
        )

    if isinstance(data, (list, tuple)) and data and isinstance(data[0], torch.Tensor):
        return data[0].cpu()

    raise TypeError(
        "Loaded object is not a torch.Tensor. Got type: " f"{type(data)}"
    )

def analyze_tensor(tensor: torch.Tensor, sample_rows: int) -> None:
    print(f"Tensor dtype: {tensor.dtype}")
    print(f"Tensor shape: {tuple(tensor.shape)}")

    total_abs_sum = tensor.abs().sum().item()
    print(f"Sum of absolute values: {total_abs_sum:.6f}")

    if total_abs_sum == 0:
        print("All values are zero.")
    else:
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        nonzero_count = torch.count_nonzero(tensor).item()
        print(f"Min value: {min_val:.6f}")
        print(f"Max value: {max_val:.6f}")
        print(f"Non-zero entries: {nonzero_count}")

        if tensor.ndim >= 2:
            row_sums = tensor.sum(dim=1)
            print("Row sum stats -> min: {:.6f}, mean: {:.6f}, max: {:.6f}".format(
                row_sums.min().item(), row_sums.mean().item(), row_sums.max().item()
            ))

    if sample_rows > 0:
        rows_to_show = min(sample_rows, tensor.shape[0])
        print(f"\nFirst {rows_to_show} rows:")
        print(tensor[:rows_to_show])

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mask_signals tensor saved to disk.")
    parser.add_argument("path",default="../tmpvis_files/orchids/mask_signals_debug.pt",help="Path to mask_signals_debug.pt")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="If the saved file is a dict, extract tensor under this key (e.g. mask_signals).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of initial rows to display (default: 5)",
    )

    args = parser.parse_args()

    try:
        tensor = load_tensor(args.path, key=args.key)
    except (FileNotFoundError, TypeError, RuntimeError, KeyError) as exc:
        print(f"Error loading tensor: {exc}")
        sys.exit(1)

    analyze_tensor(tensor, args.sample_rows)

if __name__ == "__main__":
    main()
