# Common helpers. Sourced by every stage script: `source "$(dirname "$0")/lib.sh"`.
set -u
B="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # bundle root
source "$B/config.env"
BIN="$B/build"
mkdir -p "$OUT_DIR"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# stage_begin NAME ; ... ; stage_end NAME ok|fail|skip "note"
stage_begin() {
  STAGE="$1"; SDIR="$OUT_DIR/$1"; mkdir -p "$SDIR"
  STAGE_T0=$(date +%s)
  echo "== stage $1 begin $(date -Is)" | tee -a "$OUT_DIR/run.log"
}
stage_end() {
  local st="$2" note="${3:-}"
  python3 - "$SDIR/status.json" "$1" "$st" "$STAGE_T0" "$note" <<'EOF'
import json, sys, time
p, name, st, t0, note = sys.argv[1:]
json.dump({"stage": name, "status": st, "started_unix": int(t0),
           "ended_unix": int(time.time()), "note": note}, open(p, "w"), indent=1)
EOF
  echo "== stage $1 end status=$st $(date -Is) $note" | tee -a "$OUT_DIR/run.log"
}

# NUMA node of a physical GPU index (empty if unknown).
numa_of() {
  local bus
  bus=$(nvidia-smi -i "$1" --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-Z' 'a-z')
  cat "/sys/bus/pci/devices/${bus#0000}/numa_node" 2>/dev/null ||
    cat "/sys/bus/pci/devices/$bus/numa_node" 2>/dev/null || true
}

# numactl prefix for a GPU, or nothing if numactl is missing / binding is not allowed.
numa_prefix() {
  local n; n=$(numa_of "$1")
  [ -n "$n" ] && [ "$n" -ge 0 ] 2>/dev/null || return 0
  command -v numactl >/dev/null || return 0
  numactl --cpunodebind="$n" --membind="$n" true 2>/dev/null &&
    echo "numactl --cpunodebind=$n --membind=$n"
}

# Restore geometry flags. 70b: Llama-3.3-70B (80 layers), 8b: Llama-3.1-8B (32 layers);
# both 8 KV heads x 128 dim, BF16, 32768 tokens -> 10 GiB / 4 GiB per file, 512 KiB items.
geo_flags() {
  case "$1" in
    70b) echo "--layers 80 --kv-heads 8 --head-dim 128 --tokens 32768" ;;
    8b)  echo "--layers 32 --kv-heads 8 --head-dim 128 --tokens 32768" ;;
    *) echo "unknown geometry $1" >&2; return 1 ;;
  esac
}

# Make sure the 8 restore files for a geometry exist (written once, reused by later stages).
ensure_files() {
  local geo=$1 i g
  mkdir -p "$DATA_DIR"
  for ((i = 0; i < NGPU; i++)); do
    local f="$DATA_DIR/${geo}_$i.bin"
    [ -f "$f.ok" ] && continue
    g=${GPU_ARR[$i]}
    CUDA_VISIBLE_DEVICES=$g "$BIN/restore" --file "$f" --write $(geo_flags "$geo") \
      >> "$OUT_DIR/files_written.log" 2>&1 || { echo "writing $f failed" >&2; return 1; }
    touch "$f.ok"
  done
}

# Device-level I/O sampler (/proc/diskstats every 50 ms) for the whole stage; for the storage
# course report. start_sampler; ... ; stop_sampler
start_sampler() {
  python3 "$B/src/iotrace.py" "$SDIR/diskstats.csv" 0.05 & SAMPLER_PID=$!
}
stop_sampler() {
  [ -n "${SAMPLER_PID:-}" ] && kill "$SAMPLER_PID" 2>/dev/null && wait "$SAMPLER_PID" 2>/dev/null
  SAMPLER_PID=
}
