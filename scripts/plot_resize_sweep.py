"""Draw the accuracy and latency measurements recorded in the KITTI 2015 sweep."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "results/kitti15-resize-sweep.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
    for ax, column, label in zip(axes, ("full_ms", "epe_px"), ("End-to-end latency (ms)", "Disparity error, EPE (px)")):
        for encoder, color, name, marker in (
            ("vits", "#00786f", "ViT-S", "o"),
            ("vitl", "#ab4b28", "ViT-L", "s"),
        ):
            series = sorted((r for r in rows if r["encoder"] == encoder), key=lambda r: float(r["da_factor"]))
            ax.plot(
                [float(r["da_factor"]) for r in series],
                [float(r[column]) for r in series],
                label=name, marker=marker, color=color, linewidth=2.2, markersize=5.5,
            )
        ax.set_title(label, loc="left", fontweight="bold", fontsize=13, pad=12)
        ax.set_xlabel("Depth Anything input scale")
        ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5])
        ax.grid(axis="y", alpha=0.24)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="best")
    fig.savefig(root / "assets/kitti15-resize-sweep.png", dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    main()
