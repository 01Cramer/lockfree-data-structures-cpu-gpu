#!/usr/bin/env bash
#
# Builds the harnesses.
#
#   scripts/build.sh                 build/       energy compiled out
#   scripts/build.sh --rapl          build-rapl/  energy on   (server)
#   scripts/build.sh --both          both         (the WSL check, see below)
#   scripts/build.sh --tsan          build-tsan/  correctness tests under TSan
#   scripts/build.sh --clean --rapl  wipe the directory first
#
# Two build directories rather than one toggled knob, because they are not
# interchangeable and mixing them wastes a rebuild every time you switch.
#
# On WSL use --both: the plain build is the one that RUNS (WSL exposes no
# powercap sysfs, so an ENABLE_RAPL binary aborts at construction by design),
# and the -rapl build exists only to prove the energy path still COMPILES.
# That path is never exercised on the development machine, so a compile check
# is the only feedback available before it reaches the server.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

need cmake "Install it, or load the module on the server."
need g++ "Install a C++20 compiler."

WANT_PLAIN=1
WANT_RAPL=0
WANT_TSAN=0
CLEAN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --rapl)  WANT_PLAIN=0; WANT_RAPL=1 ;;
    --both)  WANT_PLAIN=1; WANT_RAPL=1 ;;
    --tsan)  WANT_PLAIN=0; WANT_TSAN=1 ;;
    --clean) CLEAN=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

# nproc-1 so an interactive session on a shared box stays usable while linking.
JOBS="$(( $(nproc 2>/dev/null || echo 2) - 1 ))"
[ "${JOBS}" -lt 1 ] && JOBS=1

configure_and_build() {
  local dir="$1"; shift
  local label="$1"; shift

  say "${label} -> ${dir}"
  if [ "${CLEAN}" -eq 1 ]; then
    wipe "${dir}"
  fi

  # Configure; on failure, wipe once and try again.
  #
  # The cause this exists for: the repo lives on a Windows drive and gets built
  # from more than one place. CMake records the source path in the cache and
  # refuses a directory configured under a different spelling of it --
  # C:/Users/... from Windows, /mnt/c/Users/... from WSL, /c/Users/... from git
  # bash. Same files, three names, and the error reads like corruption.
  #
  # Detected by behaviour rather than by comparing the recorded path to this
  # shell's idea of it, because with three spellings in play that comparison
  # false-positives and would wipe a perfectly good tree on every other run.
  # A build directory whose configure fails is unusable and entirely
  # regenerable, so a wipe costs only the rebuild.
  if ! cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/${dir}" \
             -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null 2>&1; then
    warn "configure failed in ${dir}/ -- wiping it and retrying once"
    rm -rf "${REPO_ROOT}/${dir}"
    cmake -S "${REPO_ROOT}" -B "${REPO_ROOT}/${dir}" \
          -DCMAKE_BUILD_TYPE=Release "$@" \
      || die "configure failed for ${dir}"
  fi

  cmake --build "${REPO_ROOT}/${dir}" -j "${JOBS}" \
    || die "build failed for ${dir}"

  ok "${dir}"
}

if [ "${WANT_PLAIN}" -eq 1 ]; then
  configure_and_build build "energy compiled out" -DENABLE_RAPL=OFF
fi

if [ "${WANT_RAPL}" -eq 1 ]; then
  if [ "$(uname -s)" != "Linux" ]; then
    die "ENABLE_RAPL is Linux-only; CMake refuses it elsewhere on purpose."
  fi
  configure_and_build build-rapl "energy on" -DENABLE_RAPL=ON
fi

if [ "${WANT_TSAN}" -eq 1 ]; then
  # Benchmarks off: TSan changes timing by an order of magnitude, so a harness
  # built under it could only ever produce misleading numbers.
  configure_and_build build-tsan "correctness tests under ThreadSanitizer" \
      -DENABLE_TSAN=ON -DENABLE_BENCHMARKS=OFF
fi

# Not built by CMake -- it answers a question about the harness's assumptions
# rather than being part of it, and it has to be runnable under taskset on a
# machine where nothing else is configured yet.
if [ "${WANT_PLAIN}" -eq 1 ] || [ "${WANT_RAPL}" -eq 1 ]; then
  outdir="${REPO_ROOT}/build"
  [ -d "${outdir}" ] || outdir="${REPO_ROOT}/build-rapl"
  g++ -std=c++20 -O0 -o "${outdir}/cpu_topology_probe" \
      "${REPO_ROOT}/scripts/cpu_topology_probe.cpp" -pthread \
    && ok "cpu_topology_probe -> ${outdir#"${REPO_ROOT}/"}/" \
    || warn "cpu_topology_probe failed to build (not fatal)"
fi

say "done"
