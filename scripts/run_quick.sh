#!/usr/bin/env bash
#
# Smoke test. BENCH_QUICK=1 cuts the budget to 1000 ops/thread and 2
# repetitions -- nothing it prints is a result.
#
#   scripts/run_quick.sh [build-dir]     default: build-rapl if usable, else build
#
# It answers three questions before the real sweep commits hours to them:
#
#   1. Does everything run to completion on this host, over the whole ladder?
#   2. Is perf_running_frac 1.0? Below that the NMI watchdog is holding a PMC
#      and every hardware counter is an under-count.
#   3. Roughly how long will the sweep take?
#
# Treat (3) as a FLOOR, not an estimate. Quick mode scales the op budget but
# not the fixed costs -- pool allocation, thread spawn, and for the list the
# prefill, which is keyRange/2 = 1024 inserts whatever the budget. So quick
# mode spends proportionally more time on setup than the sweep will, and the
# list's extrapolation is the least trustworthy of the three.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${1:-}"
if [ -z "${BUILD_DIR}" ]; then
  if [ -x "${REPO_ROOT}/build-rapl/bench_stack" ] && rapl_readable; then
    BUILD_DIR="${REPO_ROOT}/build-rapl"
  else
    BUILD_DIR="${REPO_ROOT}/build"
  fi
else
  BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
fi
[ -x "${BUILD_DIR}/bench_stack" ] || die "no harnesses in ${BUILD_DIR} -- run scripts/build.sh"

OUT="${RESULTS_DIR}/quick"
mkdir -p "${OUT}"

fds="$(raise_fd_limit)"
export BENCH_QUICK=1
export BENCH_PERF=1

if is_measurement_host; then
  export BENCH_PIN=1
  pin_note="pinned"
else
  pin_note="UNPINNED (not the measurement host; BENCH_PIN would abort)"
fi

say "quick run"
printf '  build      %s\n' "${BUILD_DIR#"${REPO_ROOT}/"}"
printf '  placement  %s\n' "${pin_note}"
printf '  fd limit   %s\n' "${fds}"
printf '  ladder     %s\n' "$(nproc --all) CPUs -> top of ladder is the largest power of two inside $(( $(nproc --all) * 2 ))"
echo

declare -A ELAPSED
for adt in stack queue list; do
  say "bench_${adt}"
  start="$(date +%s)"
  if "${BUILD_DIR}/bench_${adt}" \
        --benchmark_out="${OUT}/bench_${adt}.json" \
        --benchmark_out_format=json \
        > "${OUT}/bench_${adt}.log" 2>&1; then
    ELAPSED["${adt}"]=$(( $(date +%s) - start ))
    ok "$(printf '%3ds' "${ELAPSED[${adt}]}")  -> ${OUT#"${REPO_ROOT}/"}/bench_${adt}.json"
  else
    ELAPSED["${adt}"]=-1
    bad "bench_${adt} failed -- tail of its log:"
    tail -20 "${OUT}/bench_${adt}.log" | sed 's/^/      /'
  fi
done

# The budget multipliers between quick mode and the sweep, per harness. Quick
# mode is 1000 ops/thread and 2 reps (scaledBudget/repetitions in variants.hpp);
# the sweep is 500000 for the bags, 10000 for the list, and 5 reps.
declare -A MULT=( [stack]=1250 [queue]=1250 [list]=25 )

echo
say "diagnostics"
if command -v python3 >/dev/null 2>&1; then
  python3 - "${OUT}" <<'PY'
import glob, json, sys, collections

out = sys.argv[1]
seen = collections.defaultdict(set)
rows = 0
for path in sorted(glob.glob(f"{out}/bench_*.json")):
    try:
        data = json.load(open(path))
    except Exception as exc:
        print(f"  could not read {path}: {exc}")
        continue
    for run in data.get("benchmarks", []):
        if run.get("run_type") != "iteration":
            continue
        rows += 1
        for key in ("perf_running_frac", "energy_monotonic",
                    "ops_ineffective_frac", "thread_time_spread"):
            if key in run:
                seen[key].add(round(float(run[key]), 3))

print(f"  {rows} iteration rows")
if not seen:
    print("  no diagnostic counters found -- BENCH_PERF may not have opened;"
          " check the .log files for the perf_event_open warning")
for key, values in sorted(seen.items()):
    lo, hi = min(values), max(values)
    span = f"{lo}" if lo == hi else f"{lo} .. {hi}"
    note = ""
    if key == "perf_running_frac" and lo < 1.0:
        note = "  <-- PMU was time-sliced: sysctl kernel.nmi_watchdog=0"
    if key == "energy_monotonic" and lo < 1.0:
        note = "  <-- a RAPL counter went backwards; drop those rows"
    if key == "ops_ineffective_frac" and hi > 0.001:
        note = "  <-- bags should be 0; for the list ~half the update share"
    print(f"  {key:<22} {span}{note}")
PY
else
  warn "no python3 -- skipping the counter check"
fi

echo
say "timing"
total_floor=0
for adt in stack queue list; do
  t="${ELAPSED[${adt}]}"
  if [ "${t}" -lt 0 ]; then
    printf '  %-6s failed\n' "${adt}"
    continue
  fi
  floor=$(( t * MULT[${adt}] ))
  total_floor=$(( total_floor + floor ))
  printf '  %-6s %4ds quick  ->  sweep floor ~%dh %dm  (x%s)\n' \
      "${adt}" "${t}" "$(( floor / 3600 ))" "$(( floor % 3600 / 60 ))" "${MULT[${adt}]}"
done
printf '  %-6s              ->  sweep floor ~%dh %dm total\n' \
    "" "$(( total_floor / 3600 ))" "$(( total_floor % 3600 / 60 ))"

echo
say "the number to watch is bench_list at T=128 on the hand-over-hand variants."
say "if that is hours rather than tens of minutes, halve kKeyRange to 1024"
say "(still 4 nodes per thread) before committing to the sweep."
