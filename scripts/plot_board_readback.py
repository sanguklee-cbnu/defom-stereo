"""Render the captured board tensors and the independent replay for the same input."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np

from verify_board_readback import ROOT, read_capture


def draw_row(fig, grid, start, title, actual, reference, actual_label):
    heading = fig.add_subplot(grid[start, :])
    heading.axis("off")
    heading.text(0, 0.1, title, fontsize=15, fontweight="semibold", color="#132f40")

    error = np.abs(actual - reference)
    shared_scale = Normalize(vmin=0, vmax=max(actual.max(), reference.max()))
    error_scale = PowerNorm(gamma=0.45, vmin=0, vmax=error.max())
    panels = [
        (actual, actual_label, "viridis", shared_scale),
        (reference, "Independent RTL arithmetic replay", "viridis", shared_scale),
        (error, "Absolute difference", "magma", error_scale),
    ]
    for column, (values, label, cmap, scale) in enumerate(panels):
        axis = fig.add_subplot(grid[start + 1, column])
        image = axis.imshow(values, cmap=cmap, norm=scale, interpolation="nearest")
        axis.set_title(label, fontsize=10.5, loc="left", pad=6)
        axis.axis("off")
        if column == 0:
            bar = fig.colorbar(image, cax=fig.add_subplot(grid[start + 2, :2]), orientation="horizontal")
            bar.set_label("Disparity (px)  ·  shared linear scale", fontsize=9)
        elif column == 2:
            bar = fig.colorbar(image, cax=fig.add_subplot(grid[start + 2, 2]), orientation="horizontal")
            ticks = error.max() * np.array([0, 0.05, 0.25, 0.5, 1.0])
            bar.set_ticks(ticks, labels=[f"{value:.2f}" for value in ticks])
            bar.set_label("Absolute difference (px)  ·  power scale (γ = 0.45)", fontsize=9)


def main():
    images = read_capture()
    fig = plt.figure(figsize=(13.0, 5.8), layout="constrained", facecolor="white")
    grid = fig.add_gridspec(6, 3, height_ratios=[0.22, 1, 0.065, 0.34, 1, 0.065])
    draw_row(
        fig, grid, 0, "FPGA low-resolution disparity   ·   160 × 40",
        images["board_q8"].astype(np.float64) / 256,
        images["reference_q8"].astype(np.float64) / 256, "Board readback (Q8 → px)",
    )
    draw_row(
        fig, grid, 3, "Host convex upsampling   ·   1280 × 320",
        images["host_prediction"].astype(np.float64),
        images["reference_prediction"].astype(np.float64), "CPU output from board tensors",
    )
    fig.savefig(ROOT / "assets/fpga-board-readback.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
