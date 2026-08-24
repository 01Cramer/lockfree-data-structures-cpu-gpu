# Scripts

Run them in this order. Each is safe to re-run.

| # | script | where | root | what |
|---|---|---|---|---|
| 1 | `host_prepare.sh` | server | yes | Fixes the clock, frees a PMC, opens RAPL. Once per boot |
| 2 | `build.sh` | both | no | `build/` and/or `build-rapl/` |
| 3 | `run_tests.sh` | both | no | Correctness. Seconds, and a broken structure invalidates everything |
| 4 | `host_check.sh` | server | no | Verifies 1 took; writes `provenance.txt` |
| 5 | `run_quick.sh` | both | no | Smoke test + the timings the budgets depend on |
| 6 | `run_sweep.sh` | server | no | The measurement. Calls 4 itself |
| 7 | `analyze.sh` | anywhere | no | Parse + plot + trust check |

```sh
# server, from a clean boot
scripts/host_prepare.sh
scripts/build.sh --rapl
scripts/run_tests.sh build-rapl
scripts/run_quick.sh
tmux new -s sweep          # a dropped SSH session takes the run with it
scripts/run_sweep.sh
scripts/analyze.sh results/<timestamp>
```

```sh
# WSL — development
scripts/build.sh --both    # plain build runs; -rapl only proves it compiles
scripts/run_tests.sh
scripts/run_quick.sh
```

## What differs between the two hosts

**RAPL does not exist in WSL** (no powercap passthrough), so an `ENABLE_RAPL`
binary aborts at construction there — deliberately, since a silently empty
energy column is only discovered after a sweep. `--both` exists so the energy
path at least gets compiled before it reaches the server, which is the only
feedback available for code that never executes locally.

**Pinning refuses to run off the measurement host.** `BENCH_PIN=1` aborts
unless the machine has 72 CPUs with siblings numbered after cores, because the
identity map `worker i -> CPU i` only describes the placement we want on that
enumeration. `run_quick.sh` and `run_sweep.sh` detect this and fall back to
unpinned rather than aborting.

## After transferring from Windows

```sh
chmod +x scripts/*.sh
```

`.gitattributes` pins `*.sh` and `*.py` to LF, because this clone has
`core.autocrlf=true` and a CRLF shebang fails as `bad interpreter:
/usr/bin/env bash^M` — an error that does not name its own cause. If you copy
by zip or scp rather than through git, check with `file scripts/*.sh`.

## Undoing the tuning

`host_prepare.sh` writes `scripts/.host_restore.sh` with the previous values
before it changes anything. The settings are machine-global and this box is
shared, so put them back when you are done — or reboot, which clears all of
them.
