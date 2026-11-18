"""Recompute the comparison on scenes held out of the 32-scene adaptation."""
import csv
from math import fsum
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1] / "results" / "kitti-int8-per-image.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    selected = [row for row in rows if row["backend_calibration_seen"] == "False"]
    pixels = [int(row["valid_gt_pixels"]) for row in selected]
    total_pixels = sum(pixels)
    print(f"{len(selected)} of {len(rows)} scenes, {total_pixels:,} valid disparity pixels")
    print(f"{'Configuration':<18} {'EPE (px)':>10} {'Out3 (%)':>10} {'D1 (%)':>10}")
    for column, label in (
        ("fp32", "FP32"),
        ("before_int8", "INT8 before"),
        ("adapted_int8", "INT8 adapted"),
    ):
        epe = fsum(float(row[f"{column}_epe_px"]) for row in selected) / len(selected)
        rates = []
        for metric in ("out3", "d1"):
            weighted = fsum(
                count * float(row[f"{column}_{metric}_percent"])
                for row, count in zip(selected, pixels)
            )
            rates.append(weighted / total_pixels)
        print(f"{label:<18} {epe:10.6f} {rates[0]:10.6f} {rates[1]:10.6f}")


if __name__ == "__main__":
    main()
