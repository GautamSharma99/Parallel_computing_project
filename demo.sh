#!/usr/bin/env bash
# Live demo for the presentation (CPU part). Press Enter to move to the next step.
#
#   bash demo.sh                      8K image (default)
#   IMG=images/4k.pgm bash demo.sh    another size
set -euo pipefail
cd "$(dirname "$0")"
IMG="${IMG:-images/8k.pgm}"
REPS="${REPS:-3}"
NCPU=$(getconf _NPROCESSORS_ONLN)
mkdir -p out

B=$'\e[1m'; D=$'\e[2m'; O=$'\e[38;5;208m'; G=$'\e[32m'; X=$'\e[0m'
step()  { echo; echo "${B}${O}━━ $1${X}"; echo; }
pause() { echo; read -r -p "${D}   [Enter] next step${X} " _ || true; }
ms_of() { cut -d, -f8; }   # last CSV field = total time in ms

# One result line: label, time, speedup vs serial, and a bar (shorter = faster).
show() {
  awk -v l="$1" -v ms="$2" -v ref="${SERIAL:-$2}" 'BEGIN {
    n = int(50 * ms / ref + 0.5); if (n < 1) n = 1; bar = ""
    for (i = 0; i < n; i++) bar = bar "█"
    printf "   %-24s %8.1f ms   %5.1fx   %s\n", l, ms, ref / ms, bar }'
}

# ---- build if needed (falls back to the macOS 26 SDK if the default one is broken)
if [ ! -x serial ] || [ ! -x openmp ]; then
  if ! make >/dev/null 2>&1; then
    export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
    make >/dev/null
  fi
fi
[ -f "$IMG" ] || { echo "Missing $IMG - run: python3 tools.py prepare"; exit 1; }
read -r W H < <(head -2 "$IMG" | tail -1)

step "1. The machine and the image"
if [ "$(uname)" = Darwin ]; then
  echo "   CPU:   $(sysctl -n machdep.cpu.brand_string), $NCPU cores" \
       "($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo ?) performance +" \
       "$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo ?) efficiency)"
else
  echo "   CPU:   $NCPU cores"
fi
awk -v w="$W" -v h="$H" 'BEGIN { printf "   Image: %d x %d = %.1f million pixels\n", w, h, w * h / 1e6 }'
echo "   Task:  5x5 Gaussian blur, then 3x3 Sobel edge detection (best of $REPS runs)"
pause

step "2. Serial version: one core"
SERIAL=$(./serial "$IMG" out/demo_serial.pgm "$REPS" | ms_of)
show "serial (1 core)" "$SERIAL"
pause

step "3. OpenMP: same code + one pragma, more threads"
for t in 1 2 4 8 "$NCPU"; do
  [ "$t" -gt "$NCPU" ] && continue
  ms=$(OMP_NUM_THREADS=$t OMP_SCHEDULE=dynamic ./openmp "$IMG" out/demo_openmp.pgm "$REPS" | ms_of)
  show "OpenMP, $t thread$([ "$t" -gt 1 ] && echo s)" "$ms"
done
echo
echo "   ${D}Speedup flattens after 4 threads: only 4 cores are fast performance cores.${X}"
pause

step "4. Scheduling: static vs dynamic ($NCPU threads)"
for s in static dynamic; do
  ms=$(OMP_NUM_THREADS=$NCPU OMP_SCHEDULE=$s ./openmp "$IMG" "out/demo_$s.pgm" "$REPS" | ms_of)
  show "schedule($s)" "$ms"
done
echo
echo "   ${D}static: equal rows per thread, fast cores wait for slow ones (load imbalance).${X}"
echo "   ${D}dynamic: rows handed out on demand, fast cores take more.${X}"
pause

step "5. Correctness: is the parallel output identical?"
for f in out/demo_openmp.pgm out/demo_static.pgm out/demo_dynamic.pgm; do
  if cmp -s out/demo_serial.pgm "$f"; then
    echo "   ${G}✓ identical${X}   $f  vs  out/demo_serial.pgm"
  else
    echo "   ✗ DIFFERENT  $f"
  fi
done
pause

step "6. The result"
if command -v sips >/dev/null && command -v open >/dev/null; then
  sips -s format png "$IMG" --out out/demo_input.png >/dev/null
  sips -s format png out/demo_openmp.pgm --out out/demo_edges.png >/dev/null
  open out/demo_input.png out/demo_edges.png
  echo "   Opened the input and the edge map in Preview."
else
  echo "   Output written to out/demo_openmp.pgm"
fi
echo
echo "   ${B}GPU part: switch to the Colab notebook, section 10 (Live demo).${X}"
echo
