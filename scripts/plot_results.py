#!/usr/bin/env python3
"""Plot the sweep. One figure per factor, matching how the sweep was designed.

    python scripts/plot_results.py results/tidy.csv -o results/figures

The sweep is one factor at a time around a centre point, so the charts fall out
of it directly: throughput against thread count, then throughput against each
of the other factors at the centre thread count. The centre is not hardcoded --
it is recovered from the data as the configuration that carries the ladder.

Statistics come from Google Benchmark's own aggregate rows, not from a second
reduction here: the line is its `median`, the band its `stddev`. It already
reduces the repetitions and reports the result, including for custom counters,
so recomputing was reimplementing the library.

The band changed with that. It used to be min/max, which is not comparable
across sample sizes -- expected range grows with n, so a 2-repetition smoke run
and a 5-repetition sweep would produce bands that differ for a reason unrelated
to the code. A standard deviation does not have that property.

The per-repetition rows stay in the CSV, for finding out whether a wide band is
five noisy runs or four clean ones and a co-tenant. They are just not reduced
a second time.

Absolute throughput only. No ratios against a baseline: there is no unsynchronized
variant in this study to divide by, and dividing by another synchronized one
would make the choice of denominator look like a result.
"""

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import ScalarFormatter
except ImportError:
    sys.exit("this script needs pandas and matplotlib: pip install pandas matplotlib")


MILLION = 1e6


def cells(frame):
    """One row per measured cell, taken from Google Benchmark's aggregates.

    Google Benchmark reduces the repetitions itself and reports the result as
    extra rows -- `aggregate_name` in {mean, median, stddev, cv} -- carrying the
    custom counters as well as its own timings. This joins the median rows to
    the stddev rows and returns a single frame, each column's spread available
    as a `<column>_sd` companion.

    Joined on `run_name`, which identifies a benchmark together with its thread
    count, so it is exactly the granularity of one cell.
    """
    aggregate = frame[frame["run_type"] == "aggregate"]
    if aggregate.empty:
        sys.exit(
            "no aggregate rows in this file. Google Benchmark emits them only "
            "for --benchmark_repetitions > 1, and the harness registers 5 (2 "
            "under BENCH_QUICK), so this was produced some other way."
        )

    median = aggregate[aggregate["aggregate_name"] == "median"]
    if median.empty:
        sys.exit("aggregate rows are present, but none of them is a median.")

    # Two sweeps parsed into one CSV would silently pick one of each pair.
    duplicated = median["run_name"].duplicated()
    if duplicated.any():
        sys.exit(
            f"{int(duplicated.sum())} cell(s) appear more than once, e.g. "
            f"{median.loc[duplicated, 'run_name'].iloc[0]}. More than one "
            f"sweep was parsed into this CSV; plot them separately."
        )

    stddev = aggregate[aggregate["aggregate_name"] == "stddev"]
    if stddev.empty:
        return median

    spread = stddev.select_dtypes("number").columns
    renamed = {name: f"{name}_sd" for name in spread}
    return median.merge(
        stddev[["run_name", *spread]].rename(columns=renamed),
        on="run_name",
        how="left",
    )


def summarize(frame, group_columns, column="ops_per_sec", scale=MILLION):
    """Select a column and its band. Computes no statistics.

    The band is the median plus and minus one standard deviation, both as
    Google Benchmark reported them. Pairing a robust centre with a non-robust
    spread is deliberate rather than an oversight: the line answers "what is
    the typical value" and the band answers "how noisy was it", which are
    different questions. It is a dispersion indicator, not a confidence
    interval, and at five repetitions it must not be read as one.
    """
    spread_column = f"{column}_sd"
    deviation = frame[spread_column] if spread_column in frame.columns else 0.0

    data = frame[list(group_columns)].copy()
    data["median"] = frame[column] / scale
    data["low"] = ((frame[column] - deviation) / scale).clip(lower=0)
    data["high"] = (frame[column] + deviation) / scale
    return data


def centre_point(frame, adt):
    """Recover the centre configuration from the shape of the data.

    Read from the *layout* sweep, because that is the only sweep registered at
    exactly one (mix, key_range): the three padded cells are measured at the
    centre config and nowhere else, so whatever config the non-NoPad rows carry
    IS the centre by construction.

    The previous rule -- 'the config spanning the most thread counts is the
    centre, and the thread count carrying the most mixes is the centre thread
    count' -- inferred it from the ladder instead, and the grid restructure broke
    it silently. All three mixes now span the full ladder, so the first test is a
    three-way tie and idxmax returns the lowest mix; and every thread count now
    carries all three mixes, so the second returns 1. Both would have produced
    plausible plots of the wrong slice.

    Centre thread count is the host's logical CPU count, not inferred: it is a
    real landmark (every SMT sibling busy, nothing oversubscribed) and it is
    recorded on every row.

    ops_per_thread is not part of any key here, and no longer needs to be. It is
    constant within a harness now that the list's key-range sweep is gone -- that
    sweep sat at its own thread count and carried its own per-range budgets, so
    its 128-key point collided with the ladder's eight-thread rung on every column
    the plots group by, and the scaling curve merged two different budgets there.
    """
    padded = frame[frame["layout"] != "NoPad"]
    configs = padded[["mix_pct", "key_range"]].drop_duplicates()
    if len(configs) != 1:
        sys.exit(
            f"{adt}: expected the layout sweep to sit at exactly one "
            f"(mix, key_range), found {len(configs)}. Either the sweep was run "
            f"with --benchmark_filter, or the harness no longer registers the "
            f"padded layouts only at the centre."
        )
    mix, keys = configs.iloc[0]

    cpus = frame.get("host_cpus")
    cpus = cpus.dropna() if cpus is not None else None
    if cpus is None or not len(cpus):
        sys.exit(f"{adt}: host_cpus is missing, so the centre thread count is unknown.")

    # Snapped to a rung that actually exists. The ladder is powers of two and
    # therefore lands on none of this host's landmarks -- 18, 36, 72 are drawn
    # as vertical lines, never measured -- so taking host_cpus verbatim gives 72,
    # which no row carries. Every filter on it would then return nothing, and
    # the failure is silent: plot_layout skips with "only one layout at the
    # centre point" and plot_by_factor quietly falls back to another slice.
    #
    # The highest rung at or below host_cpus is 64 here: maximum contention
    # without oversubscription, which is what the layout figure wants.
    ladder = sorted(int(t) for t in frame["threads"].dropna().unique())
    if not ladder:
        sys.exit(f"{adt}: no thread counts in the data.")
    at_or_below = [t for t in ladder if t <= int(cpus.iloc[0])]
    threads = at_or_below[-1] if at_or_below else ladder[-1]

    return {"mix_pct": mix, "key_range": keys, "threads": threads}


def host_boundaries(frame):
    """The thread counts where the measurement host changes regime.

    The ladder is powers of two, so it lands on none of these; they are drawn as
    vertical lines instead. Read from the columns the harness records rather than
    hardcoded here, so a figure cannot end up annotated for the wrong machine.
    Empty for a smoke-test file from a dev box, which has no such columns.
    """
    columns = ("host_physical_cores", "host_sockets", "host_cpus")
    if any(frame.get(name) is None or not len(frame[name].dropna())
           for name in columns):
        return []
    cores, sockets, cpus = (int(frame[name].dropna().iloc[0]) for name in columns)
    return [
        (cores // sockets, ":", "one socket"),
        (cores, "-.", "all cores"),
        (cpus, "--", "hardware threads"),
    ]


def plot_scaling(frame, centre, adt, output):
    subset = frame[
        (frame["layout"] == "NoPad")
        & (frame["mix_pct"] == centre["mix_pct"])
        & (frame["key_range"] == centre["key_range"])
    ]
    if subset.empty:
        print(f"skipped {adt}_scaling.png: no NoPad rows at the centre point")
        return

    data = summarize(subset, ["impl", "threads"])
    figure, axes = plt.subplots(figsize=(9, 5.5))

    for impl, rows in data.groupby("impl"):
        rows = rows.sort_values("threads")
        axes.plot(rows["threads"], rows["median"], marker="o", ms=3, label=impl)
        axes.fill_between(rows["threads"], rows["low"], rows["high"], alpha=0.15)

    # The ladder doubles, so a linear axis crushes everything below 16 into the
    # left margin.
    axes.set_xscale("log", base=2)
    axes.set_xticks(sorted(data["threads"].unique()))
    axes.get_xaxis().set_major_formatter(ScalarFormatter())

    # The ladder is powers of two, so it does not land on this machine's
    # structural boundaries -- they are drawn instead, because a bend in a curve
    # means something different in each region:
    #
    #   .. cores/socket   private core each, one socket
    #   .. total cores    both sockets; a coherence miss may cross the interconnect
    #   .. hardware thr.  SMT: two workers share one core's L1/L2 and issue ports
    #   beyond            oversubscribed; a lock holder can be descheduled
    for value, style, text in host_boundaries(frame):
        axes.axvline(value, color="grey", ls=style, lw=1, label=f"{text} ({value})")

    axes.set_xlabel("threads")
    axes.set_ylabel("throughput (M ops/s)")
    axes.set_title(
        f"{adt}: scaling at {centre['mix_pct']:.0f}% "
        f"{'find' if adt == 'list' else 'push'}, NoPad"
    )
    axes.legend()
    axes.grid(alpha=0.3)
    save(figure, output / f"{adt}_scaling.png")


def plot_cost(frame, centre, adt, output):
    """Throughput per CPU-second beside throughput per wall-second.

    The pair is the point. A variant that converts waiting into spinning buys
    wall-clock throughput with core-seconds, and only the second panel charges
    it for them -- so a curve that rises on the left while falling on the right
    is a variant winning by consuming more of the machine, not by doing more
    work. Plotted together because either one alone invites the wrong reading.
    """
    if "ops_per_cpu_sec" not in frame.columns:
        print(f"skipped {adt}_cost.png: no CPU-time columns in this sweep")
        return

    subset = frame[
        (frame["layout"] == "NoPad")
        & (frame["mix_pct"] == centre["mix_pct"])
        & (frame["key_range"] == centre["key_range"])
    ].dropna(subset=["ops_per_cpu_sec"])
    if subset.empty:
        print(f"skipped {adt}_cost.png: no usable CPU-time rows at the centre")
        return

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    panels = [
        ("ops_per_sec", "throughput (M ops/s of wall time)"),
        ("ops_per_cpu_sec", "efficiency (M ops/s of CPU time)"),
    ]
    for axis, (column, label) in zip(axes, panels):
        data = summarize(subset, ["impl", "threads"], column)
        for impl, rows in data.groupby("impl"):
            rows = rows.sort_values("threads")
            axis.plot(rows["threads"], rows["median"], marker="o", ms=3, label=impl)
            axis.fill_between(rows["threads"], rows["low"], rows["high"], alpha=0.15)
        axis.set_xlabel("threads")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    # Unlabelled here: the scaling figure's legend already names them, and this
    # figure has two panels sharing one legend that is already full of variants.
    for value, style, _ in host_boundaries(frame):
        for axis in axes:
            axis.axvline(value, color="grey", ls=style, lw=1)

    axes[0].legend()
    figure.suptitle(
        f"{adt}: the same work divided by wall time and by CPU time, "
        f"at {centre['mix_pct']:.0f}%, NoPad"
    )
    save(figure, output / f"{adt}_cost.png")


def plot_latency(frame, centre, adt, output):
    """Latency percentiles against thread count, one panel each.

    Log scale, because the interesting result is a tail that departs from the
    median by orders of magnitude rather than by a factor. The p50 panel should
    look much like the throughput chart inverted; where the p99.9 panel does not
    is where a progress guarantee is or is not doing anything.
    """
    percentiles = [
        ("latency_p50_ns", "p50"),
        ("latency_p99_ns", "p99"),
        ("latency_p999_ns", "p99.9"),
    ]
    available = [(c, n) for c, n in percentiles if c in frame.columns]
    if not available:
        print(f"skipped {adt}_latency.png: run with BENCH_LATENCY=1 to produce it")
        return

    subset = frame[
        (frame["layout"] == "NoPad")
        & (frame["mix_pct"] == centre["mix_pct"])
        & (frame["key_range"] == centre["key_range"])
    ].dropna(subset=[available[0][0]])
    if subset.empty:
        print(f"skipped {adt}_latency.png: no latency rows at the centre point")
        return

    figure, axes = plt.subplots(
        1, len(available), figsize=(4.6 * len(available), 5.2), sharex=True
    )
    axes = axes if len(available) > 1 else [axes]

    for axis, (column, name) in zip(axes, available):
        data = summarize(subset, ["impl", "threads"], column, scale=1.0)
        for impl, rows in data.groupby("impl"):
            rows = rows.sort_values("threads")
            axis.plot(rows["threads"], rows["median"], marker="o", ms=3, label=impl)
            axis.fill_between(rows["threads"], rows["low"], rows["high"], alpha=0.15)
        axis.set_yscale("log")
        axis.set_xlabel("threads")
        axis.set_title(name)
        axis.grid(alpha=0.3, which="both")

    axes[0].set_ylabel("latency (ns, log scale)")
    axes[0].legend()

    figure.suptitle(
        f"{adt}: operation latency at {centre['mix_pct']:.0f}%, NoPad. "
        f"Every operation measured; raw, see latency_overhead_ns."
    )
    save(figure, output / f"{adt}_latency.png")


def plot_by_factor(frame, centre, adt, column, label, output, filename):
    padded = frame[frame["layout"] == "NoPad"]
    if padded[column].nunique() < 2:
        print(f"skipped {filename}: {column} does not vary")
        return

    # Each factor is swept at whichever thread count it was registered at -- mix
    # now varies at every thread count in the main grid, while the list's
    # key-range sweep runs only at the lower count the coarse-locked variants can
    # survive. So prefer the centre, and fall back to wherever the factor varies.
    #
    # The fallback used to be the whole rule, and the grid restructure turned it
    # into a silent wrong-slice bug: with mix varying at all nine thread counts,
    # every count ties at three and idxmax returns the lowest, so the mix figure
    # would have been drawn at 1 thread -- uncontended, and the least interesting
    # slice on the plot.
    counts = padded.groupby("threads")[column].nunique()
    if counts.max() < 2:
        print(f"skipped {filename}: no single thread count sweeps {column}")
        return
    centre_threads = centre["threads"]
    threads = (
        int(centre_threads)
        if counts.get(centre_threads, 0) >= 2
        else int(counts.idxmax())
    )

    subset = padded[padded["threads"] == threads]
    data = summarize(subset, ["impl", column])
    pivot = data.pivot(index=column, columns="impl", values="median")

    figure, axes = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=axes, width=0.8)
    axes.set_xlabel(label)
    axes.set_ylabel("throughput (M ops/s)")
    axes.set_title(f"{adt}: {label} at {threads} threads, NoPad")
    axes.grid(alpha=0.3, axis="y")
    save(figure, output / filename)


def plot_layout(frame, centre, adt, output):
    subset = frame[frame["threads"] == centre["threads"]]
    subset = subset[subset["mix_pct"] == centre["mix_pct"]]
    if subset["layout"].nunique() < 2:
        print(f"skipped {adt}_layout.png: only one layout at the centre point")
        return

    data = summarize(subset, ["impl", "layout"])
    pivot = data.pivot(index="layout", columns="impl", values="median")

    figure, axes = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=axes, width=0.8)
    axes.set_xlabel("")
    axes.set_ylabel("throughput (M ops/s)")
    axes.set_title(
        f"{adt}: cache-line layout at {centre['threads']} threads.\n"
        f"NoPad is the reference; the others are treatments against it."
    )
    axes.grid(alpha=0.3, axis="y")
    plt.setp(axes.get_xticklabels(), rotation=20, ha="right")
    save(figure, output / f"{adt}_layout.png")


def plot_energy(frame, centre, adt, output):
    # The _sd exclusion is load-bearing: cells() adds a standard-deviation
    # companion for every numeric column, and "nj_per_op_package-0_sd" also
    # starts with the domain prefix. Without it each domain gets a second
    # figure plotting its own spread as though it were energy.
    domains = [
        c for c in frame.columns
        if c.startswith("nj_per_op_") and not c.endswith("_sd")
    ]
    if not domains:
        return

    # Both guards drop a row rather than correcting it. energy_window_ok is 0
    # when the window was shorter than the RAPL refresh; energy_monotonic is 0
    # when a counter went backwards, which at these window lengths means a
    # foreign write to the rw energy_uj file, not a wrap. On an aggregate row
    # these are medians of the per-repetition flags, so anything below 1 means
    # at least half the repetitions were affected.
    subset = frame[
        (frame["layout"] == "NoPad")
        & (frame["mix_pct"] == centre["mix_pct"])
        & (frame.get("energy_window_ok", 1) == 1)
        & (frame.get("energy_monotonic", 1) == 1)
    ]
    if subset.empty:
        return

    # Domains are never summed: core and uncore are subsets of their package
    # and psys overlaps it, so one figure per domain and the reader chooses.
    for domain in domains:
        figure, axes = plt.subplots(figsize=(9, 5.5))
        for impl, rows in subset.groupby("impl"):
            rows = rows.sort_values("threads")
            axes.plot(rows["threads"], rows[domain], marker="o", ms=3, label=impl)
        axes.set_xlabel("threads")
        axes.set_ylabel("energy per operation (nJ)")
        axes.set_title(
            f"{adt}: {domain.replace('nj_per_op_', '')}\n"
            f"RAPL is an activity model, and the window includes machine idle draw."
        )
        axes.legend()
        axes.grid(alpha=0.3)
        save(figure, output / f"{adt}_energy_{domain.replace('nj_per_op_', '')}.png")


def save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="output of parse_results.py")
    parser.add_argument("-o", "--output", default="results/figures")
    arguments = parser.parse_args()

    frame = cells(pd.read_csv(arguments.csv))

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    for adt, rows in frame.groupby("adt"):
        centre = centre_point(rows, adt)
        print(f"{adt}: centre point {centre}")

        plot_scaling(rows, centre, adt, output)
        plot_by_factor(
            rows,
            centre,
            adt,
            "mix_pct",
            "find %" if adt == "list" else "push %",
            output,
            f"{adt}_mix.png",
        )
        if adt == "list":
            plot_by_factor(
                rows, centre, adt, "key_range", "key range", output, "list_keys.png"
            )
        plot_layout(rows, centre, adt, output)
        plot_cost(rows, centre, adt, output)
        plot_latency(rows, centre, adt, output)
        plot_energy(rows, centre, adt, output)


if __name__ == "__main__":
    main()
