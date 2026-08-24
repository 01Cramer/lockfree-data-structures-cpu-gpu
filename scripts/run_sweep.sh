#!/usr/bin/env bash
#
# The measurement run.
#
#   scripts/run_sweep.sh                 all three, into results/<timestamp>/
#   scripts/run_sweep.sh stack queue     a subset
#   scripts/run_sweep.sh --force list    run even if host_check.sh fails
#   scripts/run_sweep.sh --no-pin        placement left to the scheduler
#   scripts/run_sweep.sh --no-perf       skip the hardware counters
#
# Harnesses run STRICTLY ONE AT A TIME. Two of them at once would contend for
# the same cores, the same L3 and the same package power budget, and every
# number from both would be wrong -- including the energy, which RAPL reports
# per socket for whatever was running.
#
# Order is stack, queue, list: cheapest first, so a problem surfaces early and
# the expensive one can be abandoned without losing the rest.
#
# Results, provenance and per-harness logs go into one timestamped directory,
# so a run is self-describing and two runs cannot overwrite each other.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FORCE=0
WANT_PIN=1
WANT_PERF=1
ADTS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --force)   FORCE=1 ;;
    --no-pin)  WANT_PIN=0 ;;
    --no-perf) WANT_PERF=0 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    stack|queue|list) ADTS+=("$1") ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done
[ "${#ADTS[@]}" -eq 0 ] && ADTS=(stack queue list)

# A dropped SSH session takes the run with it. Six hours in, that is the whole
# afternoon, so this is worth a sentence rather than a footnote.
if [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ] && [ -n "${SSH_CONNECTION:-}" ]; then
  warn "not inside tmux or screen, over SSH. If the connection drops, so does"
  warn "the sweep. Start one first:  tmux new -s sweep"
  read -r -p '  continue anyway? [y/N] ' reply
  case "${reply}" in [yY]*) ;; *) die "aborted" ;; esac
fi

if [ -x "${REPO_ROOT}/build-rapl/bench_stack" ] && rapl_readable; then
  BUILD_DIR="${REPO_ROOT}/build-rapl"
  energy_note="on"
else
  BUILD_DIR="${REPO_ROOT}/build"
  energy_note="off (no readable RAPL, or no build-rapl/)"
fi
[ -x "${BUILD_DIR}/bench_stack" ] || die "no harnesses in ${BUILD_DIR} -- run scripts/build.sh --rapl"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "${OUT}"

say "pre-flight"
if "${REPO_ROOT}/scripts/host_check.sh" "${OUT}/provenance.txt"; then
  :
elif [ "${FORCE}" -eq 1 ]; then
  warn "host_check failed and --force was given; the conditions are recorded"
  warn "in provenance.txt, so at least the results will say what they are."
else
  die "host_check failed. Fix it, or re-run with --force if you meant it."
fi

fds="$(raise_fd_limit)"
if [ "${WANT_PERF}" -eq 1 ]; then export BENCH_PERF=1; else unset BENCH_PERF; fi

if [ "${WANT_PIN}" -eq 1 ] && is_measurement_host; then
  export BENCH_PIN=1
  pin_note="pinned, worker i to CPU i"
else
  unset BENCH_PIN
  pin_note="unpinned"
  [ "${WANT_PIN}" -eq 1 ] && warn "not the measurement host, so running unpinned"
fi

echo
say "sweep ${RUN_ID}"
printf '  harnesses  %s\n' "${ADTS[*]}"
printf '  build      %s\n' "${BUILD_DIR#"${REPO_ROOT}/"}"
printf '  placement  %s\n' "${pin_note}"
printf '  energy     %s\n' "${energy_note}"
printf '  hw perf    %s\n' "$([ "${WANT_PERF}" -eq 1 ] && echo on || echo off)"
printf '  fd limit   %s\n' "${fds}"
printf '  output     %s\n' "${OUT#"${REPO_ROOT}/"}"
echo

sweep_start="$(date +%s)"
FAILED=()

for adt in "${ADTS[@]}"; do
  say "bench_${adt}  (started $(date +%H:%M:%S))"
  start="$(date +%s)"

  # stdbuf so the log is followable with tail -f during a long run rather than
  # arriving in 4 KB lumps.
  if stdbuf -oL -eL "${BUILD_DIR}/bench_${adt}" \
        --benchmark_out="${OUT}/bench_${adt}.json" \
        --benchmark_out_format=json \
        > "${OUT}/bench_${adt}.log" 2>&1; then
    elapsed=$(( $(date +%s) - start ))
    ok "$(printf '%dh %02dm %02ds' $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) $(( elapsed % 60 )))"
  else
    FAILED+=("${adt}")
    bad "bench_${adt} failed after $(( ($(date +%s) - start) / 60 ))m -- tail of its log:"
    tail -20 "${OUT}/bench_${adt}.log" | sed 's/^/      /'
    # Kept going rather than aborting: the harnesses are independent, and a
    # structural check failing in one says nothing about the others.
  fi
done

total=$(( $(date +%s) - sweep_start ))

# Re-read the machine afterwards. If a co-tenant arrived mid-sweep, the load
# average here is the only record that it happened.
{
  echo
  echo "# After the sweep, $(date -Is)"
  echo "loadavg   $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "memory    $(free -g | awk '/^Mem:/ {print $3" of "$2" GB used"}')"
  echo "duration  $(( total / 3600 ))h $(( total % 3600 / 60 ))m"
} >> "${OUT}/provenance.txt"

echo
say "finished in $(( total / 3600 ))h $(( total % 3600 / 60 ))m"
if [ "${#FAILED[@]}" -gt 0 ]; then
  bad "failed: ${FAILED[*]}"
fi
say "next:  scripts/analyze.sh ${OUT#"${REPO_ROOT}/"}"
