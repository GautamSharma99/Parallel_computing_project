#!/usr/bin/env bash
# Runs every version on every image in images/ and saves the timings as CSV.
#
#   bash bench.sh cpu     serial + OpenMP (1, 2, 4, ... threads; static and dynamic)
#   bash bench.sh gpu     CUDA naive + tiled
#   bash bench.sh all     everything
#
# Optional settings:  REPS=5  THREADS="1 2 4 8"  BLOCKS="8 16 32"
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-all}"
case "$MODE" in cpu|gpu|all) ;; *) echo "usage: bash bench.sh [cpu|gpu|all]" >&2; exit 1 ;; esac
REPS="${REPS:-5}"
BLOCKS="${BLOCKS:-16}"

# Thread counts: powers of two up to the number of cores, plus the core count.
NCPU=$(getconf _NPROCESSORS_ONLN)
if [ -z "${THREADS:-}" ]; then
  THREADS=""
  t=1
  while [ "$t" -lt "$NCPU" ]; do THREADS="$THREADS $t"; t=$((t * 2)); done
  THREADS="$THREADS $NCPU"
fi

shopt -s nullglob
images=(images/*.pgm)
if [ ${#images[@]} -eq 0 ]; then
  echo "No images in images/ - run: python3 tools.py prepare" >&2
  exit 1
fi

mkdir -p results out
CSV="results/$MODE.csv"
echo "version,threads,width,height,blur_ms,sobel_ms,transfer_ms,total_ms" > "$CSV"

for img in "${images[@]}"; do
  name=$(basename "$img" .pgm)
  echo "=== $name" >&2
  if [ "$MODE" != gpu ]; then
    ./serial "$img" "out/${name}_serial.pgm" "$REPS" | tee -a "$CSV"
    for sched in static dynamic; do
      for t in $THREADS; do
        OMP_SCHEDULE=$sched OMP_NUM_THREADS=$t \
          ./openmp "$img" "out/${name}_openmp.pgm" "$REPS" | tee -a "$CSV"
      done
    done
  fi
  if [ "$MODE" != cpu ]; then
    for v in naive tiled; do
      for b in $BLOCKS; do
        ./cuda_filter "$img" "out/${name}_cuda_${v}.pgm" "$v" "$b" "$REPS" | tee -a "$CSV"
      done
    done
  fi
done
echo "Saved $CSV" >&2
