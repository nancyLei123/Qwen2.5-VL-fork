#!/usr/bin/env bash
set -euo pipefail

# Safely process the interrupted Longoria recording by avoiding the corrupted
# shutdown tail. Each chunk is cut, health-checked, then submitted to MPS.

ROOT="/Users/leijianan/gen1_glasses"
DATA_ROOT="$ROOT/VRS_Pre_Processing/data"
SRC_DIR="$DATA_ROOT/Longoria_clean_tailtrim"
SRC_VRS="$SRC_DIR/Longoria_clean_tailtrim.vrs"
SRC_JSON="$SRC_DIR/Longoria_clean_tailtrim.vrs.json"
OUT_ROOT="$DATA_ROOT/Longoria_chunks_validation"
VRS_TOOL="$ROOT/vrs_chunking_tools/build/vrs/tools/vrs/vrs"
ARIA_VENV="$ROOT/projectaria_tools/.venv"
ARIA_MPS="$ARIA_VENV/bin/aria_mps"
HEALTH_CHECK="$ARIA_VENV/bin/run_vrs_health_check"
PYTHON="$ARIA_VENV/bin/python"

export PATH="$ARIA_VENV/bin:$PATH"
export SSL_CERT_FILE="$("$PYTHON" -c 'import certifi; print(certifi.where())')"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

CHUNKS=(
  "Longoria_chunk01_0000_0600:0:600"
  "Longoria_chunk02_0600_1200:600:1200"
  "Longoria_chunk03_1200_1800:1200:1800"
  "Longoria_chunk04_1800_2400:1800:2400"
  "Longoria_chunk05_2400_3000:2400:3000"
  "Longoria_chunk06_3000_3600:3000:3600"
  "Longoria_chunk07_3600_3780:3600:3780"
)

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

should_run_chunk() {
  local chunk_name="$1"
  shift || true

  if [[ "$#" -eq 0 ]]; then
    return 0
  fi

  local requested
  for requested in "$@"; do
    if [[ "$requested" == "$chunk_name" || "$requested" == "${chunk_name#Longoria_chunk}" ]]; then
      return 0
    fi
  done
  return 1
}

check_health_passed() {
  local compact_json="$1"
  "$PYTHON" - "$compact_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

bad = {
    name: info.get("final_result")
    for name, info in data.items()
    if info.get("final_result") != "pass"
}
if bad:
    print(f"Health check did not fully pass: {bad}", file=sys.stderr)
    sys.exit(1)
print("Health check passed for all reported configurations.")
PY
}

require_file "$SRC_VRS"
require_file "$SRC_JSON"
require_file "$VRS_TOOL"
require_file "$ARIA_MPS"
require_file "$HEALTH_CHECK"

mkdir -p "$OUT_ROOT"

echo "Longoria chunk processing root: $OUT_ROOT"
echo "Source VRS: $SRC_VRS"
echo "Important: final corrupted shutdown tail is intentionally excluded after +3780s."

for spec in "${CHUNKS[@]}"; do
  IFS=":" read -r chunk_name start_s end_s <<< "$spec"

  if ! should_run_chunk "$chunk_name" "$@"; then
    continue
  fi

  chunk_dir="$OUT_ROOT/$chunk_name"
  chunk_vrs="$chunk_dir/$chunk_name.vrs"
  chunk_json="$chunk_dir/$chunk_name.vrs.json"
  output_dir="$chunk_dir/mps_${chunk_name}_vrs"
  slam_done="$output_dir/slam/closed_loop_trajectory.csv"
  health_json="$chunk_dir/vrs_health_check.json"
  health_compact_json="$chunk_dir/vrs_health_check_compact.json"
  health_log="$chunk_dir/vrs_health_check.log"
  mps_log="$chunk_dir/mps_submit.log"

  echo
  echo "=== $chunk_name (${start_s}s to ${end_s}s) ==="

  if [[ -f "$slam_done" ]]; then
    echo "SLAM result already exists, skipping MPS: $slam_done"
    continue
  fi

  mkdir -p "$chunk_dir"

  if [[ ! -f "$chunk_vrs" ]]; then
    echo "Cutting VRS chunk..."
    "$VRS_TOOL" copy "$SRC_VRS" --to "$chunk_vrs" --range "+$start_s" "+$end_s" --no-progress
    cp "$SRC_JSON" "$chunk_json"
  else
    echo "Chunk VRS already exists, reusing: $chunk_vrs"
  fi

  if [[ ! -f "$health_compact_json" ]]; then
    echo "Running VRS health check..."
    "$HEALTH_CHECK" \
      --path "$chunk_vrs" \
      --output-json "$health_json" \
      --output-json-compact "$health_compact_json" \
      > "$health_log" 2>&1
  else
    echo "Health check already exists, validating compact result."
  fi

  check_health_passed "$health_compact_json"

  echo "Submitting to MPS SLAM. This may take a long time for 10-minute chunks."
  "$ARIA_MPS" single \
    --input "$chunk_dir" \
    --features SLAM \
    --persist-on-failure \
    --no-ui \
    2>&1 | tee "$mps_log"

  if [[ -f "$slam_done" ]]; then
    echo "SLAM result ready: $slam_done"
  else
    echo "MPS command ended, but SLAM output was not found yet: $slam_done" >&2
    exit 1
  fi
done

echo
echo "Requested Longoria chunks are complete."
