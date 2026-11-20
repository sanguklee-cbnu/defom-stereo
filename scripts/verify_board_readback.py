"""Recompute the numerical discrepancies recorded in the October 30 FPGA run."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "results/board-sample.npz"
FULL_ATOL_PX = 1e-4

def read_capture():
    report = json.loads((ROOT / "results/board-runs.json").read_text(encoding="utf-8"))
    run = report["integrated_2026_10_30"]
    spec = {
        "board_q8": ((1, 1, 40, 160), "<i4", run["lowres_q8"]["actual_sha256"]),
        "reference_q8": ((1, 1, 40, 160), "<i4", run["lowres_q8"]["expected_sha256"]),
        "host_prediction": ((1, 1, 320, 1280), "<f4", run["host_prediction"]["actual_sha256"]),
        "reference_prediction": ((1, 1, 320, 1280), "<f4", run["host_prediction"]["expected_sha256"]),
        "board_mask_int8": ((1, 576, 40, 160), "i1", run["board_mask_int8_raw_sha256"]),
    }
    images = {}
    with np.load(CAPTURE, allow_pickle=False) as capture:
        for name, (shape, dtype, digest) in spec.items():
            tensor = capture[name]
            if tensor.shape != shape or tensor.dtype != np.dtype(dtype):
                raise ValueError(f"Unexpected shape or dtype for {name}: {tensor.shape}, {tensor.dtype}")
            if hashlib.sha256(tensor.tobytes(order="C")).hexdigest() != digest:
                raise ValueError(f"Tensor {name} differs from the recorded capture")
            if shape[1] == 1:
                images[name] = tensor[0, 0]
    return images


def check_error(actual, reference, *, tolerance, expected, unit, label):
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    count = int(np.count_nonzero(error > tolerance))
    maximum, mean = float(error.max()), float(error.mean())
    expected_count, expected_maximum, expected_mean = expected
    if count != expected_count or not np.allclose(
        [maximum, mean], [expected_maximum, expected_mean], rtol=0.0, atol=1e-10
    ):
        raise ValueError(f"Computed {label} statistics differ from the run record")
    print(f"{label}: {count:,}/{error.size:,} above {tolerance:g} {unit}; "
          f"max |difference| = {maximum:.8g}, mean = {mean:.10g} {unit}")


def main():
    capture = read_capture()
    check_error(
        capture["board_q8"], capture["reference_q8"], tolerance=0,
        expected=(6171, 961, 16.19109375), unit="Q8 counts", label="FPGA readback",
    )
    check_error(
        capture["host_prediction"], capture["reference_prediction"], tolerance=FULL_ATOL_PX,
        expected=(409465, 38.62230682373047, 0.3031613931723405),
        unit="px", label="CPU-upsampled prediction",
    )
    print("Capture and statistics verified. Original model-reference comparison: MISMATCH.")


if __name__ == "__main__":
    main()
