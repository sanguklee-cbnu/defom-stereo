"""Reconstruct the October 30 host prediction from its two FPGA readbacks.

NumPy implementation of the candidate runner's postprocess_defom.py. The
effective mask scale comes from runner/references/noskip/MANIFEST.json.
"""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_CAPTURE = Path(__file__).resolve().parents[1] / "results" / "board-sample.npz"
FACTOR = 8
MASK_EFFECTIVE_SCALE = np.float32(0.2117108022134135)
ATOL_PX = 1e-4


def upsample_disparity(board_q8, board_mask_int8):
    """Apply the captured quantization and return a float32 NCHW disparity in pixels."""
    if board_q8.ndim != 4 or board_q8.shape[1] != 1 or board_q8.dtype != np.int32:
        raise ValueError("board_q8 must be an int32 array with shape [N, 1, H, W]")
    batch, _, height, width = board_q8.shape
    mask_shape = (batch, 9 * FACTOR * FACTOR, height, width)
    if board_mask_int8.shape != mask_shape or board_mask_int8.dtype != np.int8:
        raise ValueError(f"board_mask_int8 must be an int8 array with shape {mask_shape}")

    # Each 8x8 subpixel position has its own softmax over a 3x3 neighborhood.
    logits = board_mask_int8.astype(np.float32) * MASK_EFFECTIVE_SCALE
    weights = logits.reshape(batch, 1, 9, FACTOR, FACTOR, height, width)
    weights = np.exp(weights - weights.max(axis=2, keepdims=True))
    weights /= weights.sum(axis=2, keepdims=True)

    disparity_px = board_q8.astype(np.float32) * np.float32(FACTOR / 256)
    padded = np.pad(disparity_px, ((0, 0), (0, 0), (1, 1), (1, 1)))
    neighbors = np.stack(
        [padded[:, :, y:y + height, x:x + width] for y in range(3) for x in range(3)],
        axis=2,
    )
    subpixels = (weights * neighbors[:, :, :, None, None]).sum(axis=2)
    return subpixels.transpose(0, 1, 4, 2, 5, 3).reshape(
        batch, 1, height * FACTOR, width * FACTOR
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="?", default=DEFAULT_CAPTURE)
    args = parser.parse_args()
    with np.load(args.capture, allow_pickle=False) as capture:
        prediction = upsample_disparity(capture["board_q8"], capture["board_mask_int8"])
        recorded = capture["host_prediction"]

    if recorded.shape != prediction.shape or recorded.dtype != np.float32:
        raise ValueError(f"host_prediction must be a float32 array with shape {prediction.shape}")
    error = np.abs(prediction - recorded)
    matches = int(np.count_nonzero(error <= ATOL_PX))
    print(f"Replayed host upsampling: {matches:,}/{error.size:,} pixels within {ATOL_PX:g} px")
    print(f"Maximum absolute difference from the recorded host prediction: {error.max():.9g} px")
    return 0 if matches == error.size else 1


if __name__ == "__main__":
    raise SystemExit(main())
