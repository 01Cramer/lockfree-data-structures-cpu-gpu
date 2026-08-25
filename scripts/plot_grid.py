#!/usr/bin/env python3
"""Plot the sweep into the two-grid tree the study was designed around.

    python scripts/plot_grid.py results/tidy.csv -o results/figures

The sweep has two grids and they answer different questions, so they get
different trees:

    Main/           the default layout, every mix, the whole ladder.
                    "How does each synchronisation mechanism behave as
                    contention rises, and does the read/write ratio change the
                    answer?"

    Supplementary/  every layout at the centre mix, the whole ladder.
                    "How much of the answer was false sharing rather than the
                    mechanism?"

        Main/mix50/Stack/01_throughput.png
        Supplementary/PadSyncPoints/Queue/01_throughput.png

Two more trees fall out of the same data and are cheap to produce:

    Overview/       one structure per panel, all mechanisms, centre mix. The
                    figures a reader sees first.
    Diagnostics/    the data-quality evidence. Not results; the reason to
                    believe the results.

Statistics come from Google Benchmark's own aggregate rows, never from a second
reduction here: the line is its `median`, the band its `stddev`. It already
reduces the repetitions and reports the result, including for the custom
counters, so recomputing them was reimplementing the library.

The band is a dispersion indicator, not a confidence interval. At five
repetitions it must not be read as one.

DERIVED LATENCY. This sweep carries no measured latency: the percentiles need
BENCH_LATENCY, which timestamps every operation and is a separate pass by
design, because that timestamping perturbs the throughput it shares a run with.
What is plotted instead is mean latency recovered from Little's law -- with N
threads issuing operations back to back and no think time, the mean time an
operation takes is N / throughput. That identity is exact, not an estimate, and
it is worth having. It is also only a mean: it says nothing about the tail, and
the tail is where an unfair lock and a fair one differ. Where measured columns
are present this script prefers them and says so on the figure.
"""

import argparse
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import ScalarFormatter
except ImportError:
    sys.exit("this script needs pandas, numpy and matplotlib: pip install pandas numpy matplotlib")


MILLION = 1e6

# Measured on the benchmark host with scripts/../benchmarks/energy_probe.cpp:
# 63.03 W per package with the machine idle, 94.36 W under load. Two thirds of
# the package draw is therefore static, present whether or not the benchmark
# runs, and dividing it by throughput yields a "nJ/op" that ranks variants in
# exactly the order throughput already did. Subtracting it leaves the part the
# algorithm is actually responsible for. Overridable, because it is a property
# of the host rather than of the code.
DEFAULT_IDLE_WATTS = 63.03

# One colour per mechanism, fixed across every figure in the tree, so a curve
# can be followed from one page of the thesis to the next. Greens are lock-free,
# blues are the blocking mutex family, warm colours the spinning family; within
# a family the coarse variant is solid and the fine-grained ones dashed.
IMPL_STYLE = {
    "lockfree":                ("#1b9e77", "o", "-",  2.0),
    "mutex":                   ("#3b5fc0", "s", "-",  1.6),
    "mutex_two_lock":          ("#7aa5f0", "^", "--", 1.4),
    "mutex_hand_over_hand":    ("#9ec9e8", "v", "--", 1.4),
    "spinlock":                ("#d95f02", "D", "-",  1.6),
    "spinlock_two_lock":       ("#f0913f", "^", "--", 1.4),
    "spinlock_hand_over_hand": ("#f3c46b", "v", "--", 1.4),
}
IMPL_ORDER = list(IMPL_STYLE)

ADT_DIR = {"stack": "Stack", "queue": "Queue", "list": "List"}


def mix_label(adt, mix):
    """What `mix_pct` means, which is not the same thing for both workloads.

    For the bags it is the share of operations that ADD -- so the remainder pop
    or dequeue. For the list it is the share that FIND, with insert and remove
    splitting what is left evenly to hold the set at half the key range. Writing
    "50% pop" on a stack figure would name the wrong half.
    """
    if adt == "list":
        return f"{int(mix)}% find, {(100 - int(mix)) // 2}% insert, {100 - int(mix) - (100 - int(mix)) // 2}% remove"
    verb = "enqueue" if adt == "queue" else "push"
    other = "dequeue" if adt == "queue" else "pop"
    return f"{int(mix)}% {verb}, {100 - int(mix)}% {other}"

# Drawn as vertical lines rather than measured: the ladder is powers of two and
# lands on none of them.
BOUNDARY_COLUMNS = ("host_physical_cores", "host_sockets", "host_cpus")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def cells(frame):
    """One row per measured cell, from Google Benchmark's own aggregates.

    Joins the `median` rows to the `stddev` rows on `run_name`, which identifies
    a benchmark together with its thread count and is therefore exactly the
    granularity of one cell. Every numeric column gains a `<name>_sd` companion.
    """
    aggregate = frame[frame["run_type"] == "aggregate"]
    if aggregate.empty:
        sys.exit(
            "no aggregate rows in this file. Google Benchmark emits them only "
            "for --benchmark_repetitions > 1, and the harness registers 5, so "
            "this was produced some other way."
        )

    median = aggregate[aggregate["aggregate_name"] == "median"]
    if median.empty:
        sys.exit("aggregate rows are present, but none of them is a median.")

    duplicated = median["run_name"].duplicated()
    if duplicated.any():
        sys.exit(
            f"{int(duplicated.sum())} cell(s) appear more than once, e.g. "
            f"{median.loc[duplicated, 'run_name'].iloc[0]}. More than one sweep "
            f"was parsed into this CSV; plot them separately."
        )

    stddev = aggregate[aggregate["aggregate_name"] == "stddev"]
    if stddev.empty:
        return median.copy()

    spread = stddev.select_dtypes("number").columns
    renamed = {name: f"{name}_sd" for name in spread}
    return median.merge(
        stddev[["run_name", *spread]].rename(columns=renamed),
        on="run_name",
        how="left",
    ).copy()


def energy_domains(frame):
    """RAPL domains present, excluding the `_sd` companions.

    Scanned rather than listed because which domains a host exposes is a
    property of the host. Excluding `_sd` explicitly: without it the spread
    column is mistaken for a domain and plotted as one.
    """
    return sorted(
        name[len("nj_per_op_"):]
        for name in frame.columns
        if name.startswith("nj_per_op_") and not name.endswith("_sd")
    )


def derive(frame, idle_watts):
    """Add the columns that are functions of measured ones.

    Each derived column gets an `_sd` companion propagated from the measurement
    it came from, so the bands stay meaningful. Where a derivation is a pure
    scaling of throughput the relative spread carries across unchanged.
    """
    data = frame
    ops = data["ops_per_sec"]
    # Relative spread of throughput, reused by everything derived from it.
    rel = (data["ops_per_sec_sd"] / ops) if "ops_per_sec_sd" in data else 0.0

    # Little's law. Closed system, N threads, no think time: the mean time an
    # operation takes is exactly N / throughput.
    data["latency_mean_us"] = data["threads"] / ops * 1e6
    data["latency_mean_us_sd"] = data["latency_mean_us"] * rel

    for domain in energy_domains(data):
        nj = f"nj_per_op_{domain}"
        # Implied average package power over the window: nJ/op x ops/s.
        data[f"power_{domain}"] = data[nj] * 1e-9 * ops
        if f"{nj}_sd" in data:
            data[f"power_{domain}_sd"] = data[f"{nj}_sd"] * 1e-9 * ops
        # The dynamic part: what the algorithm added over an idle package.
        data[f"marginal_{nj}"] = (data[nj] - idle_watts * 1e9 / ops).clip(lower=0)
        if f"{nj}_sd" in data:
            data[f"marginal_{nj}_sd"] = data[f"{nj}_sd"]

    if "ctx_voluntary" in data:
        data["ctx_vol_per_op"] = data["ctx_voluntary"] / data["ops_total"]
    if "ctx_involuntary" in data:
        data["ctx_invol_per_op"] = data["ctx_involuntary"] / data["ops_total"]

    # Throughput relative to the same cell's single-thread measurement. Kept
    # even though the single-thread point is the noisiest in the sweep (no
    # coherence traffic at all, so it is a different regime and the run-to-run
    # spread there is the machine's, not the algorithm's) -- the figure carries
    # that warning in its subtitle rather than being withheld.
    key = ["adt", "impl", "layout", "mix_pct"]
    base = data[data["threads"] == 1].set_index(key)["ops_per_sec"]
    joined = data.set_index(key).index.map(base)
    data["speedup"] = ops.to_numpy() / np.asarray(joined, dtype=float)
    data["speedup_sd"] = data["speedup"] * rel
    return data


# --------------------------------------------------------------------------
# figure machinery
# --------------------------------------------------------------------------

def host_boundaries(frame):
    """Thread counts where the host changes regime."""
    if any(frame.get(name) is None or not len(frame[name].dropna())
           for name in BOUNDARY_COLUMNS):
        return []
    cores, sockets, cpus = (int(frame[name].dropna().iloc[0]) for name in BOUNDARY_COLUMNS)
    return [
        (cores // sockets, ":", "one socket"),
        (cores, "-.", "all cores"),
        (cpus, "--", "hardware threads"),
    ]


def band(rows, column, scale):
    """Median and +/- one standard deviation, both as Google Benchmark gave them."""
    centre = rows[column] / scale
    spread = rows[f"{column}_sd"] / scale if f"{column}_sd" in rows else 0.0
    return centre, (centre - spread), (centre + spread)


def draw_series(axis, subset, column, scale, log_y, floor=None):
    """One curve per mechanism. Returns True if anything was drawn."""
    drawn = False
    for impl in IMPL_ORDER:
        rows = subset[subset["impl"] == impl].sort_values("threads")
        if rows.empty or rows[column].isna().all():
            continue
        colour, marker, style, width = IMPL_STYLE[impl]
        centre, low, high = band(rows, column, scale)
        # A log axis cannot show a band that reaches zero, and a standard
        # deviation wider than the median does reach it. Clipped to a small
        # fraction of the median so the band stays visible and honest rather
        # than silently dropping the point.
        if log_y:
            low = np.maximum(low, centre * 0.02)
        elif floor is not None:
            low = np.maximum(low, floor)
        axis.plot(rows["threads"], centre, color=colour, marker=marker, ms=4,
                  ls=style, lw=width, label=impl)
        axis.fill_between(rows["threads"], low, high, color=colour, alpha=0.13,
                          linewidth=0)
        drawn = True
    return drawn


def decorate(axis, subset, ladder, label, log_y, boundaries, legend=True):
    axis.set_xscale("log", base=2)
    axis.set_xticks(ladder)
    axis.get_xaxis().set_major_formatter(ScalarFormatter())
    if log_y:
        axis.set_yscale("log")
    for value, style, text in boundaries:
        axis.axvline(value, color="grey", ls=style, lw=1,
                     label=f"{text} ({value})" if legend else None)
    axis.set_xlabel("threads")
    axis.set_ylabel(label)
    axis.grid(alpha=0.3, which="both")
    if legend:
        axis.legend(fontsize=8, ncol=2)


# Metrics bounded in [0, 1] are always drawn linearly, whatever their dynamic
# range. A log axis on a fraction is actively misleading: the lock-free variants
# spend essentially none of their time in the kernel, and on a log axis that
# floor of near-zero noise is stretched across six decades and reads as
# structure. On a linear axis it correctly reads as "zero".
LINEAR_COLUMNS = frozenset({"cpu_utilization"})


def wants_log(series, column=None):
    """Log scale when the values span more than about a decade and a half."""
    if column is not None and (column in LINEAR_COLUMNS or column.endswith("_frac")):
        return False
    values = series.dropna()
    values = values[values > 0]
    if len(values) < 2:
        return False
    return float(values.max() / values.min()) > 30


def save(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


def single(subset, column, ylabel, title, subtitle, path, scale=1.0,
           log_y=None, boundaries=(), ladder=()):
    if column not in subset.columns or subset[column].isna().all():
        return False
    if log_y is None:
        log_y = wants_log(subset[column], column)
    figure, axis = plt.subplots(figsize=(8.6, 5.4))
    if not draw_series(axis, subset, column, scale, log_y):
        plt.close(figure)
        return False
    decorate(axis, subset, ladder, ylabel, log_y, boundaries)
    axis.set_title(title, fontsize=11)
    if subtitle:
        figure.text(0.5, -0.02, subtitle, ha="center", fontsize=8, color="#444")
    save(figure, path)
    return True


def paired(subset, specs, title, subtitle, path, boundaries=(), ladder=()):
    """Two metrics side by side, sharing an x axis and one legend."""
    usable = [s for s in specs if s[0] in subset.columns and not subset[s[0]].isna().all()]
    if not usable:
        return False
    figure, axes = plt.subplots(1, len(usable), figsize=(6.6 * len(usable), 5.2),
                               squeeze=False)
    axes = axes[0]
    for axis, (column, ylabel, scale) in zip(axes, usable):
        log_y = wants_log(subset[column], column)
        draw_series(axis, subset, column, scale, log_y)
        decorate(axis, subset, ladder, ylabel, log_y, boundaries, legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, fontsize=8, ncol=2)
    figure.suptitle(title, fontsize=11)
    if subtitle:
        figure.text(0.5, -0.01, subtitle, ha="center", fontsize=8, color="#444")
    save(figure, path)
    return True


# --------------------------------------------------------------------------
# the figure set produced in every leaf directory
# --------------------------------------------------------------------------

def leaf_figures(subset, frame, directory, title, idle_watts, measured_latency):
    """Every metric, for one (grid slice, structure) pair.

    Numbered so the three headline metrics sort above the supporting ones in a
    file listing: throughput, latency and energy first, then the counters that
    explain them.
    """
    boundaries = host_boundaries(frame)
    ladder = sorted(int(t) for t in subset["threads"].dropna().unique())
    written = 0
    domains = energy_domains(subset)

    def note(text):
        return text

    # --- 1. THROUGHPUT ---------------------------------------------------
    written += single(
        subset, "ops_per_sec", "throughput (M ops/s)",
        f"{title} -- throughput", None,
        directory / "01_throughput.png", MILLION, None, boundaries, ladder)

    written += single(
        subset, "speedup", "throughput relative to 1 thread",
        f"{title} -- scaling relative to a single thread",
        note("Normalised to the single-thread cell, which is the noisiest point in the "
             "sweep: with one thread there is no coherence traffic at all, so its "
             "run-to-run spread reflects the machine rather than the algorithm."),
        directory / "02_speedup.png", 1.0, None, boundaries, ladder)

    # --- 2. LATENCY ------------------------------------------------------
    if measured_latency:
        written += paired(
            subset,
            [("latency_p50_ns", "p50 latency (ns)", 1.0),
             ("latency_p99_ns", "p99 latency (ns)", 1.0)],
            f"{title} -- measured latency", None,
            directory / "03_latency.png", boundaries, ladder)
        written += single(
            subset, "latency_p999_ns", "p99.9 latency (ns)",
            f"{title} -- tail latency", None,
            directory / "04_latency_tail.png", 1.0, True, boundaries, ladder)
    else:
        written += single(
            subset, "latency_mean_us", "mean latency per operation (us)",
            f"{title} -- mean operation latency (derived)",
            note("Derived, not measured: threads / throughput, which is exact for a closed "
                 "loop with no think time (Little's law). It is a mean only. Percentiles and "
                 "the tail need a BENCH_LATENCY pass, which this sweep did not run."),
            directory / "03_latency_mean_derived.png", 1.0, None, boundaries, ladder)

    # --- 3. ENERGY -------------------------------------------------------
    for domain in domains:
        written += single(
            subset, f"nj_per_op_{domain}", f"energy per operation ({domain}, nJ)",
            f"{title} -- energy per operation, {domain}",
            note(f"Total package energy divided by operations. About two thirds of this "
                 f"package's draw is static ({idle_watts:.1f} W idle), so this figure "
                 f"largely restates throughput; see the marginal figure for the part the "
                 f"algorithm is responsible for."),
            directory / f"05_energy_{domain}.png", 1.0, None, boundaries, ladder)

        written += single(
            subset, f"marginal_nj_per_op_{domain}",
            f"dynamic energy per operation ({domain}, nJ)",
            f"{title} -- marginal energy per operation, {domain}",
            note(f"Package energy above an idle baseline of {idle_watts:.2f} W, divided by "
                 f"operations. This is the part attributable to the work rather than to the "
                 f"machine being switched on."),
            directory / f"06_energy_marginal_{domain}.png", 1.0, None, boundaries, ladder)

    power_specs = [(f"power_{d}", f"implied package power ({d}, W)", 1.0) for d in domains]
    if power_specs:
        written += paired(
            subset, power_specs,
            f"{title} -- implied average package power",
            note("Energy per operation times throughput. A flat line near the host's busy "
                 "draw is the expected shape; it is the sanity check on the energy figures "
                 "rather than a result."),
            directory / "07_power.png", boundaries, ladder)

    # --- 4. WHERE THE MACHINE WENT ---------------------------------------
    written += paired(
        subset,
        [("ops_per_sec", "throughput (M ops/s of wall time)", MILLION),
         ("ops_per_cpu_sec", "efficiency (M ops/s of CPU time)", MILLION)],
        f"{title} -- the same work per wall-second and per CPU-second",
        note("The pair is the point. A variant that converts waiting into spinning buys "
             "wall-clock throughput with core-seconds, and only the right panel charges it "
             "for them."),
        directory / "10_wall_vs_cpu_time.png", boundaries, ladder)

    written += paired(
        subset,
        [("cpu_utilization", "CPU utilisation (1.0 = every thread busy)", 1.0),
         ("cpu_sys_frac", "fraction of CPU time in the kernel", 1.0)],
        f"{title} -- how much of the machine was used, and by whom",
        note("Utilisation well below 1.0 means threads were blocked or asleep rather than "
             "running. A high kernel fraction alongside it identifies the mechanism: futex "
             "waits for the mutex, nanosleep backoff for the spinlock."),
        directory / "11_cpu_utilisation.png", boundaries, ladder)

    written += paired(
        subset,
        [("thread_time_spread", "thread time spread (slowest / fastest)", 1.0),
         ("thread_cpu_spread", "thread CPU-time spread (slowest / fastest)", 1.0)],
        f"{title} -- fairness between worker threads",
        note("1.0 is perfectly even. A mechanism that reaches high throughput by letting "
             "one thread run a long batch while the others wait shows it here, and only "
             "here: the throughput figure cannot distinguish it from genuine parallelism."),
        directory / "12_fairness.png", boundaries, ladder)

    written += paired(
        subset,
        [("ctx_vol_per_op", "voluntary context switches per operation", 1.0),
         ("ctx_invol_per_op", "involuntary context switches per operation", 1.0)],
        f"{title} -- context switches per operation",
        note("Voluntary switches are the thread giving up the CPU on purpose -- a futex "
             "wait under the mutex, a blocking nanosleep under the spinlock. The lock-free "
             "variants should show almost none of either."),
        directory / "13_context_switches.png", boundaries, ladder)

    # --- 5. HARDWARE COUNTERS --------------------------------------------
    written += paired(
        subset,
        [("ipc", "instructions per cycle", 1.0),
         ("cycles_per_op", "cycles per operation", 1.0)],
        f"{title} -- instruction throughput",
        note("IPC collapsing towards zero while cycles per operation climbs is the "
             "signature of a core retiring almost nothing: a CAS loop failing and "
             "retrying, or a lock word being pulled across the interconnect."),
        directory / "20_ipc_and_cycles.png", boundaries, ladder)

    written += paired(
        subset,
        [("stall_frac", "cycles stalled in the back end", 1.0),
         ("sb_stall_frac", "cycles stalled on a full store buffer", 1.0)],
        f"{title} -- where the cycles went",
        note("CYCLE_ACTIVITY.STALLS_TOTAL and RESOURCE_STALLS.SB as fractions of all "
             "cycles. The first is general back-end pressure; the second is specifically "
             "the store buffer draining, which is what a lock prefix and a fence cost."),
        directory / "21_stalls.png", boundaries, ladder)

    written += paired(
        subset,
        [("l1d_miss_per_op", "L1D read misses per operation", 1.0),
         ("branch_misp_per_op", "branch mispredictions per operation", 1.0)],
        f"{title} -- memory and control flow",
        note("An L1D miss on a shared line usually means another core invalidated it, so "
             "this counts coherence traffic. Mispredictions rise where a retry loop's exit "
             "becomes unpredictable."),
        directory / "22_cache_and_branches.png", boundaries, ladder)

    return written


# --------------------------------------------------------------------------
# the trees
# --------------------------------------------------------------------------

def build_main(frame, output, idle_watts, measured_latency):
    """Default layout, every mix, whole ladder."""
    written = 0
    grid = frame[frame["layout"] == "NoPad"]
    for mix in sorted(grid["mix_pct"].dropna().unique(), reverse=True):
        for adt in ("stack", "queue", "list"):
            subset = grid[(grid["adt"] == adt) & (grid["mix_pct"] == mix)]
            if subset.empty:
                continue
            directory = output / "Main" / f"mix{int(mix):02d}" / ADT_DIR[adt]
            title = f"{ADT_DIR[adt]}, NoPad, {mix_label(adt, mix)}"
            written += leaf_figures(subset, frame, directory, title,
                                    idle_watts, measured_latency)
            subset.to_csv(directory / "data.csv", index=False)
    return written


def build_supplementary(frame, output, idle_watts, measured_latency, centre_mix):
    """Every layout at the centre mix, whole ladder."""
    written = 0
    grid = frame[frame["mix_pct"] == centre_mix]
    for layout in sorted(grid["layout"].dropna().unique()):
        for adt in ("stack", "queue", "list"):
            subset = grid[(grid["adt"] == adt) & (grid["layout"] == layout)]
            if subset.empty:
                continue
            directory = output / "Supplementary" / layout / ADT_DIR[adt]
            present = sorted(subset["impl"].unique())
            title = f"{ADT_DIR[adt]}, {layout}, {mix_label(adt, centre_mix)}"
            written += leaf_figures(subset, frame, directory, title,
                                    idle_watts, measured_latency)
            subset.to_csv(directory / "data.csv", index=False)
            (directory / "variants_present.txt").write_text(
                "\n".join(present) + "\n"
                "\n# Variants absent from this layout were not measured: a padded\n"
                "# layout is bit-identical to NoPad for a structure with no lock\n"
                "# word, and the two-lock queue refuses to instantiate where an\n"
                "# unpadded tail would land on the head mutex's cache line.\n")
    return written


def build_overview(frame, output, centre_mix, idle_watts):
    """One row of panels per metric, one column per structure."""
    grid = frame[(frame["layout"] == "NoPad") & (frame["mix_pct"] == centre_mix)]
    if grid.empty:
        return 0
    boundaries = host_boundaries(frame)
    ladder = sorted(int(t) for t in grid["threads"].dropna().unique())
    domains = energy_domains(frame)
    energy_column = f"marginal_nj_per_op_{domains[0]}" if domains else None

    rows = [
        ("ops_per_sec", "throughput (M ops/s)", MILLION),
        ("latency_mean_us", "mean latency (us, derived)", 1.0),
    ]
    if energy_column:
        rows.append((energy_column, f"dynamic energy (nJ/op, {domains[0]})", 1.0))
    rows += [
        ("ops_per_cpu_sec", "efficiency (M ops/CPU-s)", MILLION),
        ("cpu_utilization", "CPU utilisation", 1.0),
        ("ipc", "instructions per cycle", 1.0),
    ]

    written = 0
    for column, ylabel, scale in rows:
        if column not in grid.columns or grid[column].isna().all():
            continue
        figure, axes = plt.subplots(1, 3, figsize=(18, 5.0), squeeze=False)
        for axis, adt in zip(axes[0], ("stack", "queue", "list")):
            subset = grid[grid["adt"] == adt]
            log_y = wants_log(subset[column], column)
            draw_series(axis, subset, column, scale, log_y)
            decorate(axis, subset, ladder, ylabel if adt == "stack" else "",
                     log_y, boundaries, legend=False)
            axis.set_title(ADT_DIR[adt])
            axis.legend(fontsize=7, ncol=1)
        figure.suptitle(f"{ylabel} -- NoPad, centre mix ({int(centre_mix)}%)", fontsize=12)
        save(figure, output / "Overview" / f"{column}.png")
        written += 1
    return written


def build_diagnostics(raw, cell_frame, output):
    """The evidence that the numbers are worth reading."""
    written = 0
    iteration = raw[raw["run_type"] == "iteration"]
    cv = raw[(raw["run_type"] == "aggregate") & (raw["aggregate_name"] == "cv")]

    # Run-to-run spread against thread count.
    if not cv.empty:
        figure, axis = plt.subplots(figsize=(8.6, 5.2))
        for adt, rows in cv.groupby("adt"):
            grouped = rows.groupby("threads")["ops_per_sec"]
            axis.plot(grouped.median().index, grouped.median().values * 100,
                      marker="o", label=f"{adt} (median)")
            axis.fill_between(grouped.quantile(0.9).index,
                              grouped.quantile(0.1).values * 100,
                              grouped.quantile(0.9).values * 100, alpha=0.12)
        axis.axhline(5, color="grey", ls="--", lw=1, label="5%")
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(cv["threads"].unique()))
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_xlabel("threads")
        axis.set_ylabel("coefficient of variation of throughput (%)")
        axis.set_title("Run-to-run spread over 5 repetitions\n"
                       "(band is the 10th-90th percentile across cells)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        save(figure, output / "Diagnostics" / "throughput_cv.png")
        written += 1

    # Load balance.
    if "thread_time_spread" in iteration:
        figure, axis = plt.subplots(figsize=(8.6, 5.2))
        for adt, rows in iteration.groupby("adt"):
            grouped = rows.groupby("threads")["thread_time_spread"]
            axis.plot(grouped.median().index, grouped.median().values,
                      marker="o", label=f"{adt} (median)")
            axis.fill_between(grouped.quantile(0.95).index,
                              grouped.quantile(0.05).values,
                              grouped.quantile(0.95).values, alpha=0.12)
        axis.axhline(1.5, color="grey", ls="--", lw=1, label="1.5 (flagged)")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(sorted(iteration["threads"].unique()))
        axis.get_xaxis().set_major_formatter(ScalarFormatter())
        axis.set_xlabel("threads")
        axis.set_ylabel("thread time spread (slowest / fastest)")
        axis.set_title("Load balance between workers\n"
                       "(band is the 5th-95th percentile across runs)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, which="both")
        save(figure, output / "Diagnostics" / "thread_time_spread.png")
        written += 1

    # Energy against throughput: the static-power problem, drawn.
    domains = energy_domains(cell_frame)
    if domains:
        column = f"nj_per_op_{domains[0]}"
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
        usable = cell_frame.dropna(subset=[column])
        usable = usable[usable["threads"] >= 4]
        for axis, (col, label) in zip(axes, [
                (column, f"energy per operation ({domains[0]}, nJ)"),
                (f"marginal_{column}", f"dynamic energy per operation ({domains[0]}, nJ)")]):
            if col not in usable:
                continue
            for adt, rows in usable.groupby("adt"):
                axis.scatter(rows["ops_per_sec"], rows[col], s=12, alpha=0.6, label=adt)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("throughput (ops/s)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3, which="both")
        axes[0].legend(fontsize=8)
        figure.suptitle("Energy per operation against throughput, 4 threads and above\n"
                        "A straight line of slope -1 means the metric is a restatement "
                        "of throughput", fontsize=11)
        save(figure, output / "Diagnostics" / "energy_vs_throughput.png")
        written += 1

    return written


README = """\
# Figures

Produced by `scripts/plot_grid.py` from the parsed sweep. Every line is Google
Benchmark's own `median` over the five repetitions; every band is its `stddev`.
The band is a dispersion indicator, not a confidence interval -- at five
repetitions it must not be read as one.

## Trees

    Main/mix<NN>/<Structure>/     default layout (NoPad), one directory per
                                  read/write mix, whole thread ladder.
    Supplementary/<Layout>/<Structure>/
                                  one directory per cache-line layout, at the
                                  centre mix, whole thread ladder.
    Overview/                     all three structures side by side, one figure
                                  per metric. The pages a reader sees first.
    Diagnostics/                  data-quality evidence, not results.

Each leaf also carries `data.csv`: exactly the rows plotted, for building tables.

## Figures in each leaf

Headline metrics first, then the counters that explain them.

    01_throughput             operations per second of wall time
    02_speedup                the same, relative to one thread
    03_latency*               per-operation latency
    05_energy_*               energy per operation, per RAPL package
    06_energy_marginal_*      the same with the idle baseline removed
    07_power                  implied average package power (a sanity check)

    10_wall_vs_cpu_time       throughput beside efficiency per CPU-second
    11_cpu_utilisation        how much of the machine ran, and how much in kernel
    12_fairness               spread between the slowest and fastest worker
    13_context_switches       voluntary and involuntary, per operation

    20_ipc_and_cycles         instructions per cycle, cycles per operation
    21_stalls                 back-end stalls and store-buffer stalls
    22_cache_and_branches     L1D read misses and branch mispredictions

## Two cautions carried on the figures themselves

`03_latency_mean_derived` is derived from Little's law (threads / throughput),
which is exact for this closed-loop harness but is a **mean**. The percentiles
and the tail need a `BENCH_LATENCY` pass, which this sweep did not run.

`05_energy_*` divides total package energy by operations, and roughly two thirds
of this host's package draw is static. That metric therefore ranks variants in
almost exactly the order throughput already did. `06_energy_marginal_*` removes
the idle baseline and is the figure to reason from.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="tidy.csv from parse_results.py")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="directory to write the figure tree into")
    parser.add_argument("--idle-watts", type=float, default=DEFAULT_IDLE_WATTS,
                        help="per-package idle draw subtracted for the marginal "
                             f"energy figures (default {DEFAULT_IDLE_WATTS}, measured "
                             "by benchmarks/energy_probe.cpp)")
    parser.add_argument("--centre-mix", type=float, default=None,
                        help="mix the supplementary grid sits at; inferred if omitted")
    arguments = parser.parse_args()

    raw = pd.read_csv(arguments.csv)
    frame = derive(cells(raw), arguments.idle_watts)

    # The centre mix is recovered rather than assumed: the padded layouts are
    # registered at exactly one mix, so whatever mix the non-NoPad rows carry
    # IS the centre by construction.
    if arguments.centre_mix is not None:
        centre_mix = arguments.centre_mix
    else:
        padded = frame[frame["layout"] != "NoPad"]["mix_pct"].dropna().unique()
        if len(padded) != 1:
            sys.exit(
                f"expected the padded layouts at exactly one mix, found {sorted(padded)}. "
                f"Pass --centre-mix to choose one.")
        centre_mix = float(padded[0])

    measured_latency = "latency_p50_ns" in frame.columns and not frame["latency_p50_ns"].isna().all()

    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    total = 0
    total += build_main(frame, output, arguments.idle_watts, measured_latency)
    total += build_supplementary(frame, output, arguments.idle_watts, measured_latency, centre_mix)
    total += build_overview(frame, output, centre_mix, arguments.idle_watts)
    total += build_diagnostics(raw, frame, output)

    (output / "README.md").write_text(README)
    frame.to_csv(output / "cells.csv", index=False)

    print(f"centre mix          {int(centre_mix)}%")
    print(f"idle baseline       {arguments.idle_watts:.2f} W per package")
    print(f"latency             {'measured' if measured_latency else 'derived (Little), no BENCH_LATENCY pass'}")
    print(f"figures written     {total}")
    print(f"output              {output}")


if __name__ == "__main__":
    main()
