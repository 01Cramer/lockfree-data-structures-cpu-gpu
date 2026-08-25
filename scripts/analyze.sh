#!/usr/bin/env bash
#
# Turns a sweep's JSON into a tidy table and figures.
#
#   scripts/analyze.sh results/20260824-141200
#   scripts/analyze.sh results/quick
#
# Needs pandas and matplotlib:  pip install --user pandas matplotlib
#
# Runs anywhere -- this is the one step that does not need the measurement
# host, so copy the results directory back and do it locally if the server has
# no python.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -eq 1 ] || die "usage: scripts/analyze.sh <results-dir>"

DIR="$1"
[ -d "${DIR}" ] || DIR="${REPO_ROOT}/$1"
[ -d "${DIR}" ] || die "no such directory: $1"

# python3 on the server and on WSL, plain python under git bash on Windows,
# where `python3` exists only as a Microsoft Store stub that prints an
# advertisement and exits 9009. Resolved rather than assumed, because analysis
# is the one step that deliberately runs away from the measurement host.
PY=""
for candidate in python3 python; do
  if command -v "${candidate}" >/dev/null 2>&1 &&
     "${candidate}" -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 else 1)' >/dev/null 2>&1; then
    PY="${candidate}"
    break
  fi
done
[ -n "${PY}" ] || die "no Python 3 on PATH. Install it, or copy the results directory to a machine that has it."

shopt -s nullglob
JSON=("${DIR}"/bench_*.json)
[ "${#JSON[@]}" -gt 0 ] || die "no bench_*.json in ${DIR}"

say "parsing ${#JSON[@]} file(s)"
# --force so re-running after a fix overwrites rather than refusing. The guard
# it disables is there to stop a benchmark dump being used as a destination,
# and the destination here is one this script chose.
"${PY}" "${REPO_ROOT}/scripts/parse_results.py" "${JSON[@]}" \
        -o "${DIR}/tidy.csv" --force \
  || die "parse_results.py failed"
ok "${DIR#"${REPO_ROOT}/"}/tidy.csv"

# Written only by a BENCH_LATENCY run, which is a separate pass -- its rows
# carry a timestamp pair per operation and must not share a table with rows
# that do not.
if [ -f "${DIR}/tidy_latency.csv" ]; then
  ok "${DIR#"${REPO_ROOT}/"}/tidy_latency.csv (latency pass)"
fi

# Two plotters over the same table, because the sweep has two grids and a flat
# directory cannot express both. plot_grid.py is the one to read from: it lays
# the figures out as Main/<mix>/<Structure>/ and Supplementary/<layout>/<Structure>/,
# which is how the results get cited. plot_results.py stays because its flat
# per-structure overview is a faster way to see whether a run is sane at all.
say "plotting (grid)"
"${PY}" "${REPO_ROOT}/scripts/plot_grid.py" "${DIR}/tidy.csv" \
        -o "${DIR}/figures" \
  || die "plot_grid.py failed"
ok "${DIR#"${REPO_ROOT}/"}/figures/"

say "plotting (flat overview)"
"${PY}" "${REPO_ROOT}/scripts/plot_results.py" "${DIR}/tidy.csv" \
        -o "${DIR}/figures/Flat" \
  || die "plot_results.py failed"
ok "${DIR#"${REPO_ROOT}/"}/figures/Flat/"

# Three columns decide whether the rest of the table can be believed, so they
# are surfaced here rather than left for someone to remember to check.
say "trust check"
"${PY}" - "${DIR}/tidy.csv" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1])
frame = frame[frame["run_type"] == "iteration"]
print(f"  {len(frame)} iteration rows")

# parse_results.py names this column run_name; `name` was never in the tidy
# table. The mistake could only ever surface on a run that had something to
# report, which is the one run where this check has to work.
LABEL = "run_name" if "run_name" in frame.columns else None

def flag(column, predicate, message):
    if column not in frame.columns:
        print(f"  {column:<22} absent")
        return
    bad = frame[predicate(frame[column])]
    if bad.empty:
        print(f"  {column:<22} clean")
        return
    share = 100.0 * len(bad) / len(frame)
    print(f"  {column:<22} {len(bad)} row(s), {share:.2f}% -- {message}")
    if LABEL:
        for label in bad[LABEL].head(3):
            print(f"      {label}")

flag("perf_running_frac", lambda c: c < 1.0,
     "PMU time-sliced, hardware counts are low")
flag("energy_monotonic", lambda c: c < 1.0,
     "RAPL went backwards, drop their joules columns")
flag("energy_window_ok", lambda c: c < 1.0,
     "window shorter than the RAPL refresh, energy is noise")
flag("thread_time_spread", lambda c: c > 1.5,
     "threads finished far apart, the window tail was under-contended")
flag("check_size", lambda c: c != frame["check_size_expected"],
     "structural mismatch -- should have aborted, investigate")
PY

echo
say "done. Figures in ${DIR#"${REPO_ROOT}/"}/figures/"
