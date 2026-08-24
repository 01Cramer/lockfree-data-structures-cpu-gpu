#!/usr/bin/env bash
#
# Correctness, before any measurement.
#
#   scripts/run_tests.sh [build-dir]     default: build
#   scripts/run_tests.sh build-tsan      the ThreadSanitizer pass
#
# The harness already validates each structure after every repetition and
# aborts on a mismatch, so this is not the only line of defence -- but that
# check fires six hours into a sweep, and this one takes seconds. A structure
# that is wrong makes every variant it is compared against meaningless too.
#
# TSan is a separate build because it changes timing by an order of magnitude:
# useful for finding a race, useless for measuring one.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BUILD_DIR="${REPO_ROOT}/${1:-build}"
[ -d "${BUILD_DIR}" ] || die "no ${BUILD_DIR} -- run scripts/build.sh first"

say "correctness tests in ${BUILD_DIR#"${REPO_ROOT}/"}"

# --output-on-failure: a passing run stays quiet, a failure prints the
# assertion rather than just the test name.
if ctest --test-dir "${BUILD_DIR}" --output-on-failure; then
  ok "all tests passed"
else
  die "tests failed -- do not measure until this is green"
fi

# Builds but is not registered with ctest: it has no assertions, it exists to
# prove every variant instantiates. Cheap to confirm it still links.
if [ -x "${BUILD_DIR}/CPU_COMPILATION" ]; then
  "${BUILD_DIR}/CPU_COMPILATION" >/dev/null && ok "CPU_COMPILATION runs"
fi
