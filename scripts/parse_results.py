#!/usr/bin/env python3
"""Turn Google Benchmark JSON into a tidy table, one row per measurement.

The harness encodes every experimental factor twice on purpose: once in the
benchmark name as key=value pairs, and once as numeric counters. The name is
what makes a result self-describing in isolation; the counters are what make
plotting possible without parsing anything. This script reconciles the two and
gives back a frame whose columns are the factors.

    python scripts/parse_results.py results/*.json -o results/tidy.csv

Rows from a BENCH_LATENCY run are written to `results/tidy_latency.csv` instead,
because their throughput and energy figures carry a timestamp pair per operation
and must not share a table with rows that do not. Existing output files are never
replaced without --force, and a benchmark dump is never a legal destination.

Rows come in two kinds, distinguished by `run_type`:

    iteration   one repetition, the raw measurement
    aggregate   Google Benchmark's own mean/median/stddev/cv over them

Both are kept. Plot from `iteration` rows and let the plotting library compute
its own intervals; the aggregate rows are there for cross-checking, not as the
primary source. Note that `cv` rows carry a ratio, not a time, so they must not
be mixed into a time column.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("this script needs pandas: pip install pandas")


# Counters the harness always emits. Anything else (the joules_* and
# nj_per_op_* columns) depends on which RAPL domains the host exposed, so it is
# picked up dynamically rather than listed here.
CORE_COUNTERS = [
    "ops_total",
    "ops_per_sec",
    "window_seconds",
    "ops_ineffective_frac",
    "thread_time_spread",
    "start_skew_ms",
    "pool_nodes_per_thread",
    "ops_per_thread",
    "mix_pct",
    "key_range",
    "host_physical_cores",
    "host_sockets",
    "check_size",
    "check_size_expected",
]

# Emitted only where the host or the run mode supplies them: the CPU-time
# columns need a working per-thread clock, the switch counts are Linux only,
# the hardware-counter columns need BENCH_PERF=1 and a readable PMU, and the
# latency columns exist only in a run with BENCH_LATENCY set. Absent columns are
# skipped rather than filled, so that "not measured" never reads as a zero.
#
# One of these is a diagnostic rather than a measurement: perf_running_frac
# below 1.0 means the counter group was time-sliced onto the PMU, so every
# other hardware column in that row is an under-count. Check it before reading
# them.
OPTIONAL_COUNTERS = [
    "ops_per_cpu_sec",
    "cpu_utilization",
    "thread_cpu_spread",
    "cpu_sys_frac",
    "ctx_voluntary",
    "ctx_involuntary",
    "cycles_per_op",
    "ipc",
    "stall_frac",
    "sb_stall_frac",
    "l1d_miss_per_op",
    "branch_misp_per_op",
    "perf_running_frac",
    "energy_monotonic",
    "latency_ops",
    "latency_p50_ns",
    "latency_p90_ns",
    "latency_p95_ns",
    "latency_p99_ns",
    "latency_p999_ns",
    "latency_mean_ns",
    "latency_stdev_ns",
    "latency_min_ns",
    "latency_max_ns",
    "latency_backwards",
    "latency_overhead_ns",
    "latency_cycles_per_ns",
    # Emitted only when the host exposed a RAPL domain.
    "energy_window_ok",
]

# Blanked on rows that carry latency_ops: everything whose value depends on how
# long an operation took, and therefore on the stamps. Counts, correctness
# checks and the factor columns are not affected, so a latency run's rows stay
# joinable to a throughput run's on (impl, layout, mix, keys, threads).
CONTAMINATED_BY_LATENCY = [
    "ops_per_sec",
    "ops_per_cpu_sec",
    "cpu_utilization",
    "cpu_sys_frac",
    "cpu_seconds",
    "thread_cpu_spread",
]

# Fields Google Benchmark owns; everything else in a benchmark entry is either
# a counter we emitted or a factor parsed out of the name.
GBENCH_FIELDS = [
    "run_name",
    "run_type",
    "aggregate_name",
    "repetitions",
    "repetition_index",
    "threads",
    "iterations",
    "real_time",
    "cpu_time",
    "time_unit",
]


def parse_name(run_name):
    """adt=queue/impl=lockfree/layout=NoPad/... -> dict.

    Google Benchmark appends '/threads:N' using a colon rather than an equals
    sign, and for aggregate rows suffixes the whole string with '_mean' and
    friends. Both are already available as dedicated JSON fields, so they are
    parsed here only to keep the segment count honest and are then discarded.
    """
    factors = {}
    for segment in run_name.split("/"):
        if "=" in segment:
            key, value = segment.split("=", 1)
            factors[key] = value
        elif ":" in segment:
            key, value = segment.split(":", 1)
            factors[key.strip("_")] = value.split("_")[0]
    return factors


def load(path):
    with open(path) as handle:
        document = json.load(handle)

    context = document.get("context", {})
    rows = []

    for entry in document.get("benchmarks", []):
        row = {"source": Path(path).name}
        row.update(parse_name(entry.get("run_name", "")))

        for field in GBENCH_FIELDS:
            if field in entry:
                row[field] = entry[field]

        # Counters arrive as top-level keys alongside the framework's own.
        for key, value in entry.items():
            if key not in GBENCH_FIELDS and key not in (
                "name",
                "family_index",
                "per_family_instance_index",
                "label",
            ):
                row.setdefault(key, value)

        row["host_cpus"] = context.get("num_cpus")
        row["host_mhz"] = context.get("mhz_per_cpu")
        row["library_build_type"] = context.get("library_build_type")
        rows.append(row)

    return rows


def write_csv(frame, path, force):
    """Write, refusing to clobber. Re-parsing is cheap; a lost dump is not."""
    if path.exists() and not force:
        sys.exit(
            f"{path} already exists. Pass --force to replace it, or name a "
            f"different file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {len(frame)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="benchmark JSON files")
    parser.add_argument(
        "-o",
        "--output",
        help="write a CSV here; rows from a BENCH_LATENCY run go to "
        "<stem>_latency<suffix> beside it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting an existing output file",
    )
    arguments = parser.parse_args()

    rows = []
    for path in arguments.inputs:
        rows.extend(load(path))

    if not rows:
        sys.exit("no benchmark entries found")

    frame = pd.DataFrame(rows)

    numeric = (
        ["threads", "mix_pct", "key_range", "ops"]
        + CORE_COUNTERS
        + OPTIONAL_COUNTERS
    )
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Latency is recorded inside the same timed region as throughput, so a
    # BENCH_LATENCY run reports both -- and its timing-derived figures carry a
    # stamp pair per operation. They are not merely a little high: the same
    # constant is most of a 25 ns lock-free operation and nothing against a
    # 6000 ns hand-over-hand one, so it compresses exactly the gap being
    # measured. Blanked rather than left plausible, because NaN is the only
    # value that cannot be plotted by accident. Latency, correctness, workload
    # composition and the factor columns are unaffected and stay.
    stamped = (
        frame["latency_ops"].notna()
        if "latency_ops" in frame
        else pd.Series(False, index=frame.index)
    )

    if stamped.any():
        contaminated = [
            column
            for column in CONTAMINATED_BY_LATENCY
            + [
                name
                for name in frame.columns
                if name.startswith("joules_") or name.startswith("nj_per_op")
            ]
            if column in frame
        ]
        if contaminated:
            frame.loc[stamped, contaminated] = float("nan")
            print(
                f"NOTE: {int(stamped.sum())} rows come from a BENCH_LATENCY "
                f"run; their throughput, cost and energy columns have been "
                f"blanked. Google Benchmark's own real_time/cpu_time fields are "
                f"left as it wrote them and are inflated for those rows too.",
                file=sys.stderr,
            )

    # Loud, because both are silent failures in a chart. A run whose window was
    # too short for the energy counters reports a joule figure that is one
    # quantum of noise; a run whose threads finished far apart did not hold its
    # nominal thread count for its own duration.
    measurements = frame[frame.get("run_type") == "iteration"]

    # The harness times the region itself and hands the figure to Google
    # Benchmark, which sums manual times across threads and divides by the
    # summed iteration count. If that reasoning is wrong, every duration in the
    # sweep is off by a factor of the thread count -- and it would look
    # perfectly plausible. window_seconds is the harness's own measurement of
    # the same interval, so the two must agree.
    if "window_seconds" in measurements and "real_time" in measurements:
        units = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}
        scale = measurements["time_unit"].map(units)
        expected = measurements["window_seconds"] * scale
        drift = (measurements["real_time"] - expected).abs() / expected
        disagreeing = measurements[drift > 0.01]
        if len(disagreeing):
            print(
                f"ERROR: {len(disagreeing)} runs where Google Benchmark's "
                f"real_time disagrees with the harness's own window by more "
                f"than 1%. Timing is not trustworthy; do not plot this.",
                file=sys.stderr,
            )

    if "energy_window_ok" in measurements:
        short = measurements[measurements["energy_window_ok"] == 0]
        if len(short):
            print(
                f"WARNING: {len(short)} runs had a timed window below the RAPL "
                f"refresh interval; their joule columns are not usable.",
                file=sys.stderr,
            )
    # Threads run equal budgets and retire at different times, so the tail of
    # the window carries fewer threads than its nominal count. That biases the
    # high-thread points optimistic. It is a disclosure, not a defect -- but it
    # has to be quantified wherever those points are used.
    if "thread_time_spread" in measurements:
        skewed = measurements[measurements["thread_time_spread"] > 1.5]
        if len(skewed):
            worst = measurements["thread_time_spread"].max()
            print(
                f"NOTE: {len(skewed)} runs had a thread time spread above 1.5 "
                f"(worst {worst:.1f}). Threads retired far apart, so the end of "
                f"those windows was less contended than the thread count "
                f"implies. Check pinning and background load.",
                file=sys.stderr,
            )

    # A late thread only lengthens the window under this metric, so this is not
    # an error -- but if the skew approaches the window, the run was nowhere
    # near as concurrent as it claims.
    if "start_skew_ms" in measurements and "window_seconds" in measurements:
        fraction = (
            measurements["start_skew_ms"] / 1e3 / measurements["window_seconds"]
        )
        late = measurements[fraction > 0.1]
        if len(late):
            print(
                f"WARNING: in {len(late)} runs the threads entered the loop "
                f"more than 10% of the window apart (worst {fraction.max():.0%}). "
                f"Those runs did not hold their nominal thread count.",
                file=sys.stderr,
            )

    # Per-thread CPU clocks are tick-accumulated on some hosts -- ~15.6 ms on
    # Windows, one scheduler tick on a kernel without fine-grained accounting.
    # Against a window of a few milliseconds that quantizes to zero or to one
    # whole tick, which produces a utilization figure that is either absent or
    # above 1.0. Both are impossible, so both are caught here rather than
    # plotted as if the clock had cooperated.
    if "cpu_utilization" in measurements:
        usable = measurements["cpu_utilization"].dropna()
        impossible = usable[(usable > 1.05) | (usable <= 0.0)]
        if len(impossible):
            print(
                f"WARNING: {len(impossible)} runs reported a CPU utilization "
                f"outside (0, 1.05] (worst {usable.max():.2f}). The per-thread "
                f"CPU clock is tick-granular on this host and the window was "
                f"too short for it. Treat every cpu_* and ops_per_cpu_sec "
                f"column in those runs as unusable, not as a measurement.",
                file=sys.stderr,
            )
    elif "ops_per_cpu_sec" not in measurements:
        print(
            "NOTE: no CPU-time columns; the host supplied no per-thread CPU "
            "clock, so throughput has no cost-side companion in this sweep.",
            file=sys.stderr,
        )

    # Every operation enters the histogram, so the only way a percentile is thin
    # is a small op budget. p99.9 over 1000 operations rests on a single one,
    # which is a draw from the tail rather than a measurement of it.
    if "latency_ops" in measurements:
        counted = measurements["latency_ops"].dropna()
        thin = counted[counted < 10000]
        if len(thin):
            print(
                f"WARNING: {len(thin)} runs recorded fewer than 10000 latency "
                f"observations (fewest {int(counted.min())}), so their p99.9 "
                f"rests on under ten. Raise the op budget before quoting that "
                f"column.",
                file=sys.stderr,
            )

        # A backwards interval means a thread moved between cores whose cycle
        # counters are not synchronized. It drops that observation, so the tail
        # is missing exactly the events a migration would have produced.
        if "latency_backwards" in measurements:
            backwards = measurements["latency_backwards"].dropna()
            affected = backwards[backwards > 0]
            if len(affected):
                print(
                    f"WARNING: {len(affected)} runs discarded backwards "
                    f"intervals (worst {int(affected.max())}), so a thread "
                    f"migrated across unsynchronized cycle counters. Re-run "
                    f"pinned (BENCH_PIN=1) before quoting the tail.",
                    file=sys.stderr,
                )
    else:
        print(
            "NOTE: latency measurement was disabled in this sweep "
            "(BENCH_LATENCY unset), so only throughput and cost are present.",
            file=sys.stderr,
        )

    if not arguments.output:
        print(frame.to_string())
        return

    target = Path(arguments.output)

    # The dumps are the only irreplaceable thing here -- a sweep costs hours, this
    # script costs a second. So it refuses to write over a benchmark JSON, or over
    # any file it was given as input, whatever --force says.
    inputs = {Path(path).resolve() for path in arguments.inputs}
    if target.resolve() in inputs or target.suffix.lower() == ".json":
        sys.exit(
            f"refusing to write to {target}: that is a benchmark dump, not a "
            f"destination for parsed output"
        )

    # A latency run's rows describe a different experiment (see the blanking
    # above), so they go to their own file rather than sharing one where a later
    # groupby could average across both.
    if stamped.any() and not stamped.all():
        write_csv(frame[~stamped], target, arguments.force)
        write_csv(
            frame[stamped],
            target.with_name(f"{target.stem}_latency{target.suffix}"),
            arguments.force,
        )
    else:
        write_csv(frame, target, arguments.force)


if __name__ == "__main__":
    main()
