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
# Usage: scan-memcheck-output.sh <pkg.Rcheck>
# Exits 1 if anything matched, 0 otherwise.

set -uo pipefail

checkdir=${1:-}
if [ -z "$checkdir" ] || [ ! -d "$checkdir" ]; then
  echo "usage: $(basename "$0") <pkg.Rcheck>" >&2
  exit 2
fi

# `ERROR SUMMARY` is matched only with a non-zero count, because valgrind
# prints it on every clean run too. The zero-size `realloc()` is listed
# separately: valgrind reports it but does not count it as an error, so the
# summary line stays at zero.
patterns='ERROR SUMMARY: [1-9][0-9]* errors|realloc\(\) with size 0|Invalid (read|write|free|realloc)|Mismatched free|Conditional jump or move depends on uninitialised|uninitialised value|AddressSanitizer|LeakSanitizer|ThreadSanitizer|runtime error:|SUMMARY: (Undefined|Address|Leak)'

found=0
while IFS= read -r f; do
  if grep -Eq "$patterns" "$f"; then
    found=1
    echo "::error file=${f}::memory-check finding in ${f#"$checkdir"/}"
    echo "---------------------------------------------------------------"
    echo "$f"
    echo "---------------------------------------------------------------"
    cat "$f"
    echo "---------------------------------------------------------------"
  fi
done < <(find "$checkdir" -type f \( -name '*.Rout' -o -name '*.Rout.fail' -o -name '*.Rout.timings' \) | sort)

if [ "$found" -eq 1 ]; then
  echo "Memory-check findings above; failing the job."
  exit 1
fi

echo "No sanitizer or valgrind findings in $checkdir."
