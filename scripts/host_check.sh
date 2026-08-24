#!/usr/bin/env bash
#
# Verifies the machine is in a fit state to measure, and records what it was.
#
#   scripts/host_check.sh [provenance-file]
#
# Two jobs, and the second matters as much as the first. The checks catch a
# forgotten host_prepare.sh before six hours of runs rather than after. The
# provenance dump is what lets the thesis say what the numbers were measured on
# -- it is written next to the results, because a results file that does not
# carry its own conditions is not reproducible by anyone, including you.
#
# Exit status: 0 if every blocking check passed. Warnings do not fail it.
# Read-only apart from the provenance file, and needs no privilege.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PROVENANCE="${1:-}"
FAILED=0
blocker() { bad "$1"; FAILED=1; }

say "host"
if is_measurement_host; then
  ok "measurement host: 72 CPUs, siblings numbered after cores"
else
  warn "NOT the measurement host ($(nproc --all) CPUs, cpu0 siblings"
  warn "'$(read_sysfs /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)')."
  warn "BENCH_PIN will abort here; the run scripts fall back to unpinned."
fi

say "tuning"
turbo_disabled && ok "no_turbo=1" \
               || blocker "no_turbo=$(read_sysfs /sys/devices/system/cpu/intel_pstate/no_turbo) -- run host_prepare.sh"
watchdog_off   && ok "nmi_watchdog=0" \
               || blocker "nmi_watchdog on: it holds a PMC, so BENCH_PERF counts come back scaled"
paranoid_ok    && ok "perf_event_paranoid=$(read_sysfs /proc/sys/kernel/perf_event_paranoid)" \
               || blocker "perf_event_paranoid=$(read_sysfs /proc/sys/kernel/perf_event_paranoid), needs <= 2"
rapl_readable  && ok "RAPL counters readable" \
               || warn "RAPL unreadable -- energy columns will be absent (fine for a non-energy run)"

governor="$(read_sysfs /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
[ "${governor}" = "performance" ] && ok "governor=performance" || warn "governor=${governor}"

hard_fd="$(ulimit -Hn)"
if [ "${hard_fd}" = "unlimited" ] || [ "${hard_fd}" -ge 4096 ]; then
  ok "fd hard limit ${hard_fd} (the run scripts raise the soft limit themselves)"
else
  warn "fd hard limit ${hard_fd}; BENCH_PERF wants 768 plus overhead at T=128"
fi

# The one confound pinning cannot protect against. A co-tenant competes for L3,
# memory bandwidth and the interconnect, and on this box RAPL will attribute
# their power draw to your window as well.
say "is the machine quiet?"
printf '  load average   %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"
printf '  memory         %s\n' "$(free -g | awk '/^Mem:/ {printf "%s GB used of %s", $3, $2}')"
echo   "  top by RSS:"
# `|| true` throughout: head closes the pipe early, which under `pipefail`
# would otherwise take the whole script down over a cosmetic listing.
ps -eo pcpu,rss,comm --sort=-rss --no-headers 2>/dev/null \
  | head -5 \
  | awk '{printf "    %6s%%cpu %8.1f GB  %s\n", $1, $2/1048576, $3}' || true

load1="$(cut -d' ' -f1 /proc/loadavg)"
if awk -v l="${load1}" 'BEGIN{exit !(l > 2.0)}'; then
  warn "load average ${load1} -- something else is running. Energy figures will"
  warn "include it (RAPL is per-socket) and cache pressure will not be visible."
fi

say "toolchain"
printf '  %s\n' "$(g++ --version | head -1)"
printf '  cmake %s\n' "$(cmake --version | head -1 | awk '{print $3}')"
printf '  kernel %s\n' "$(uname -r)"
if command -v python3 >/dev/null 2>&1; then
  python3 -c 'import pandas, matplotlib' 2>/dev/null \
    && ok "python3 with pandas and matplotlib" \
    || warn "python3 lacks pandas/matplotlib -- analyze.sh will fail (pip install pandas matplotlib)"
fi

if [ -n "${PROVENANCE}" ]; then
  mkdir -p "$(dirname "${PROVENANCE}")"
  {
    echo "# Measurement conditions, captured $(date -Is)"
    echo
    echo "host           $(hostname)"
    echo "kernel         $(uname -r)"
    echo "compiler       $(g++ --version | head -1)"
    echo "git            $(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'not a repo')$(git -C "${REPO_ROOT}" diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo
    echo "no_turbo             $(read_sysfs /sys/devices/system/cpu/intel_pstate/no_turbo)"
    echo "nmi_watchdog         $(read_sysfs /proc/sys/kernel/nmi_watchdog)"
    echo "perf_event_paranoid  $(read_sysfs /proc/sys/kernel/perf_event_paranoid)"
    echo "scaling_governor     ${governor}"
    echo "loadavg              $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "memory               $(free -g | awk '/^Mem:/ {print $3" of "$2" GB used"}')"
    echo
    echo "# Mitigations. All of them price kernel entry, and only the lock-based"
    echo "# variants enter the kernel -- so this host systematically favours the"
    echo "# lock-free side. Recorded rather than disabled."
    for f in /sys/devices/system/cpu/vulnerabilities/*; do
      if [ -e "${f}" ]; then
        printf '%-24s %s\n' "$(basename "${f}")" "$(cat "${f}")"
      fi
    done
    echo
    echo "# lscpu"
    lscpu
    echo
    echo "# lscpu -e"
    lscpu -e
  } > "${PROVENANCE}"
  ok "provenance written to ${PROVENANCE}"
fi

echo
if [ "${FAILED}" -eq 0 ]; then
  say "${C_GREEN}ready${C_OFF}"
else
  say "${C_RED}not ready${C_OFF} -- see the failures above"
fi
exit "${FAILED}"
