# Shared helpers. Sourced by the other scripts, never executed on its own.
#
# Run order, start to finish:
#
#   1. host_prepare.sh   once per boot, root. Fixes the clock and frees a PMC.
#   2. build.sh          once per code change.
#   3. run_tests.sh      correctness. A sweep of a broken structure is waste.
#   4. host_check.sh     verifies 1 took, and records what the machine was.
#   5. run_quick.sh      smoke test. Gives the timings the budgets depend on.
#   6. run_sweep.sh      the measurement.
#   7. analyze.sh        parse + plot.
#
# On WSL, 2/3/5 work and 1/4/6 do not: no powercap, no intel_pstate, and the
# topology tripwire refuses to pin on a host that is not the Xeon.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"

# Build directory names, overridable as a set:
#
#   export BENCH_BUILD_PREFIX=build-wsl
#
# Worth doing whenever two toolchains share one checkout. Visual Studio treats
# `build/` as its own working directory and keeps `build/.vs/` open, so from
# WSL those files cannot be deleted and cannot be rewritten -- a Windows lock is
# not a permission a chmod can grant. Giving each toolchain its own prefix costs
# a rebuild once and removes the collision permanently.
BUILD_PREFIX="${BENCH_BUILD_PREFIX:-build}"
BUILD_PLAIN="${BUILD_PREFIX}"
BUILD_RAPL="${BUILD_PREFIX}-rapl"
BUILD_TSAN="${BUILD_PREFIX}-tsan"

if [ -t 1 ]; then
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_RED=''; C_GREEN=''; C_YELLOW=''; C_BOLD=''; C_OFF=''
fi

say()  { printf '%s==>%s %s\n' "${C_BOLD}" "${C_OFF}" "$*"; }
ok()   { printf '  %sok%s   %s\n' "${C_GREEN}" "${C_OFF}" "$*"; }
warn() { printf '  %swarn%s %s\n' "${C_YELLOW}" "${C_OFF}" "$*" >&2; }
bad()  { printf '  %sFAIL%s %s\n' "${C_RED}" "${C_OFF}" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "${C_RED}" "${C_OFF}" "$*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is not on PATH. $2"
}

# Empty when already root, so the same line works either way.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

read_sysfs() { cat "$1" 2>/dev/null || echo "n/a"; }

# Remove a build directory, tolerating files that cannot be deleted. On a
# Windows drive, an open Visual Studio holds locks inside build/.vs that WSL
# cannot break; rm reports those and removes everything else, which includes
# CMakeCache.txt and is all that actually needs to go. Without the `|| true`
# its non-zero exit takes the whole script down under `set -e` -- after the
# delete has already succeeded.
wipe() {
  rm -rf "${REPO_ROOT:?}/${1:?}" 2>/dev/null || true
}

# The measurement host, tested exactly as benchmarks/support/topology.hpp does:
# 72 logical CPUs, and cpu0's sibling numbered 36 (so the kernel enumerated
# cores before SMT siblings, which is what makes the identity pinning map the
# placement we want). Both must hold, or BENCH_PIN aborts the run.
is_measurement_host() {
  local cpus siblings
  cpus="$(nproc --all 2>/dev/null || echo 0)"
  siblings="$(read_sysfs /sys/devices/system/cpu/cpu0/topology/thread_siblings_list)"
  [ "${cpus}" = "72" ] && [[ "${siblings}" == 0,36* ]]
}

# Whether the tuning host_prepare.sh applies is actually in force. Callers
# decide whether a false is fatal; run_sweep treats it as fatal, run_quick does
# not, because a smoke test is allowed to be untuned.
turbo_disabled() { [ "$(read_sysfs /sys/devices/system/cpu/intel_pstate/no_turbo)" = "1" ]; }
watchdog_off()   { [ "$(read_sysfs /proc/sys/kernel/nmi_watchdog)" = "0" ]; }

paranoid_ok() {
  local level
  level="$(read_sysfs /proc/sys/kernel/perf_event_paranoid)"
  case "${level}" in
    ''|*[!0-9-]*) return 1 ;;
    *) [ "${level}" -le 2 ] ;;
  esac
}

rapl_readable() {
  head -c1 /sys/class/powercap/intel-rapl:0/energy_uj >/dev/null 2>&1
}

# Six perf descriptors per worker, and the ladder tops out at 128 workers. The
# default soft limit of 1024 clears that by a margin too thin to trust once the
# JSON output and Google Benchmark's own descriptors are counted. Raising the
# soft limit needs no privilege as long as the hard limit allows it.
raise_fd_limit() {
  local wanted=4096 hard
  hard="$(ulimit -Hn)"
  if [ "${hard}" != "unlimited" ] && [ "${hard}" -lt "${wanted}" ]; then
    wanted="${hard}"
  fi
  ulimit -n "${wanted}" 2>/dev/null || true
  echo "${wanted}"
}
