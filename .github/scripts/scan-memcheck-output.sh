#!/usr/bin/env bash
#
# Scan a finished `<pkg>.Rcheck` directory for sanitizer and valgrind findings.
#
# `R CMD check` can pass while the instrumented R reports a memory error: the
# tools write to stderr, which R folds into the per-file `.Rout`, and only a
# *failing* test leaves a `.Rout.fail` behind. The r-hub containers' own
# `r-check` greps `<pkg>-Ex.Rout` and `tests/*.Rout.fail` only, so a finding in
# a test that passes goes unnoticed -- which is how the zero-size `realloc()`
# in kde1d 1.2.0 got past CI and was caught by CRAN instead. This scans every
# output file in the check directory, whether or not its test passed.
#
# Usage: scan-memcheck-output.sh [--require PATTERN] <pkg.Rcheck>
#
#   --require PATTERN  Fail unless PATTERN appears somewhere in the scanned
#                      output. A clean run and a run where the tool never
#                      started look identical -- both report nothing -- so for
#                      valgrind, which prints an `ERROR SUMMARY:` line even
#                      when it finds nothing, this turns "no findings" into
#                      positive evidence that the check really was instrumented.
#                      The sanitizers print nothing on a clean run and so have
#                      no equivalent marker; the workflow checks their
#                      toolchain for `-fsanitize` up front instead.
#
# Exits 1 on a finding or a missing --require marker, 0 otherwise.

set -uo pipefail

require=""
while [ $# -gt 0 ]; do
  case "$1" in
    --require) require=${2:-}; shift 2 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

checkdir=${1:-}
if [ -z "$checkdir" ] || [ ! -d "$checkdir" ]; then
  echo "usage: $(basename "$0") [--require PATTERN] <pkg.Rcheck>" >&2
  exit 2
fi

# `ERROR SUMMARY` is matched only with a non-zero count, because valgrind
# prints it on every clean run too. The zero-size `realloc()` is listed
# separately: valgrind reports it but does not always count it as an error.
patterns='ERROR SUMMARY: [1-9][0-9]* errors|realloc\(\) with size 0|Invalid (read|write|free|realloc)|Mismatched free|Conditional jump or move depends on uninitialised|uninitialised value|AddressSanitizer|LeakSanitizer|ThreadSanitizer|runtime error:|SUMMARY: (Undefined|Address|Leak)'

found=0
scanned=0
saw_marker=0

while IFS= read -r f; do
  scanned=$((scanned + 1))
  if [ -n "$require" ] && grep -Eq "$require" "$f"; then
    saw_marker=1
  fi
  if grep -Eq "$patterns" "$f"; then
    found=1
    echo "::error file=${f}::memory-check finding in ${f#"$checkdir"/}"
    echo "---------------------------------------------------------------"
    echo "$f"
    echo "---------------------------------------------------------------"
    cat "$f"
    echo "---------------------------------------------------------------"
  fi
done < <(find "$checkdir" -type f \( -name '*.Rout' -o -name '*.Rout.fail' \) | sort)

if [ "$scanned" -eq 0 ]; then
  echo "::error::no .Rout files under $checkdir; nothing was actually checked"
  exit 1
fi

if [ "$found" -eq 1 ]; then
  echo "Memory-check findings above; failing the job."
  exit 1
fi

if [ -n "$require" ] && [ "$saw_marker" -eq 0 ]; then
  echo "::error::none of the $scanned output files matched '$require'"
  echo "The check reported no findings, but it also shows no sign of having"
  echo "been instrumented, so a clean result here would not mean anything."
  exit 1
fi

echo "No sanitizer or valgrind findings in $checkdir ($scanned output files scanned)."
