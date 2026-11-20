"""Replay the output-channel schedules in the 2026-10-26 hybrid Conv trace.

The source is npu_conv_timing.json (SHA-256
106defed3db966a02d814768365c3938ee0054d743f5f9f2fefbcedd751a1fff).
The CSV keeps one row per executor record in capture order. A core's
lo:hi field is its half-open output-channel range. An empty field means
that core had no plan. execute_seconds measures the wall-clock execution
window shared by the started cores for that invocation.
"""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_TRACE = Path(__file__).resolve().parents[1] / "results" / "hybrid-conv-trace.csv"


def channel_shards(channels, cores):
    """Port of the parallel executor's channel_shards and 16-channel align."""
    if channels <= 0 or cores <= 0:
        raise ValueError("output channels and configured cores must be positive")
    padded = (channels + 15) // 16 * 16
    unit = 32 if padded % 32 == 0 and padded // 32 >= cores else 16
    blocks = padded // unit
    count = min(cores, blocks)
    quotient, remainder = divmod(blocks, count)
    ranges = []
    start = 0
    for selector in range(count):
        stop = start + unit * (quotient + (selector < remainder))
        ranges.append((start, min(stop, channels)))
        start = stop
    return ranges


def verify(trace):
    hardware_calls = 0
    inactive_reuses = 0
    core_starts = 0
    durations = []
    with trace.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            label = f"call {index} ({row['layer']})"
            if int(row["call_index"]) != index:
                raise ValueError(f"{label}: capture order differs")
            channels = int(row["output_channels"])
            cores = int(row["configured_cores"])
            if cores != 4 or channels <= 0:
                raise ValueError(f"{label}: invalid trace geometry")
            recorded = [row[f"core_{selector}_channels"] for selector in range(cores)]
            starts = int(row["actual_core_starts"])
            completed = int(row["fresh_completions"])
            seconds = float(row["execute_seconds"])
            if row["mode"] == "inactive_reuse":
                if any(recorded) or starts != 0 or completed != 0 or seconds != 0:
                    raise ValueError(f"{label}: inactive reuse contains a hardware event")
                inactive_reuses += 1
                continue
            if row["mode"] != "hardware":
                raise ValueError(f"{label}: unknown mode {row['mode']!r}")
            shards = channel_shards(channels, cores)
            expected = [f"{lo}:{hi}" for lo, hi in shards] + [""] * (cores - len(shards))
            if recorded != expected:
                raise ValueError(f"{label}: observed {recorded}, expected {expected}")
            if starts != len(shards) or completed != starts:
                raise ValueError(f"{label}: starts or fresh completions differ from schedule")
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"{label}: invalid execution time {seconds!r}")
            hardware_calls += 1
            core_starts += starts
            durations.append(seconds)

    if (hardware_calls, inactive_reuses, core_starts) != (604, 51, 2299):
        raise ValueError(f"unexpected capture totals: {hardware_calls=}, {inactive_reuses=}, {core_starts=}")
    total_seconds = math.fsum(durations)
    if not math.isclose(total_seconds, 1.338172432500869, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"unexpected summed execution time: {total_seconds:.15f} s")
    print(f"{hardware_calls + inactive_reuses} executor records: {hardware_calls} hardware, "
          f"{inactive_reuses} inactive reuses")
    print(f"{core_starts:,} physical starts and fresh completions; every channel schedule matched")
    print(f"Summed Conv execute time: {total_seconds:.9f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, default=DEFAULT_TRACE)
    args = parser.parse_args()
    try:
        verify(args.trace)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Trace verification failed: {exc}\n")
