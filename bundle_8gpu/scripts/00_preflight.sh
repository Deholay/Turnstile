#!/bin/bash
# Stage 00: record the machine and check prerequisites. Never modifies the system.
source "$(dirname "$0")/lib.sh"
stage_begin 00_preflight
E="$SDIR"
fail=()

{
  echo "date: $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "os: $(. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME")"
  echo "gpus_configured: $GPUS"
} > "$E/summary.txt"
nvidia-smi > "$E/nvidia-smi.txt" 2>&1 || fail+=("nvidia-smi failed")
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id,compute_cap \
  --format=csv > "$E/gpus.csv" 2>&1
nvidia-smi topo -m > "$E/topo.txt" 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv > "$E/gpu_busy.csv" 2>&1
lscpu > "$E/lscpu.txt" 2>&1
numactl -H > "$E/numa.txt" 2>&1 || echo "numactl not installed" > "$E/numa.txt"
free -g > "$E/mem.txt" 2>&1
lsblk -o NAME,SIZE,TYPE,MODEL,ROTA,MOUNTPOINT > "$E/lsblk.txt" 2>&1
cat /proc/mdstat > "$E/mdstat.txt" 2>/dev/null
for d in /sys/block/nvme*n*; do
  [ -e "$d" ] || continue
  echo "$(basename "$d") model=$(cat "$d/device/model" 2>/dev/null | xargs) numa=$(cat "$d/device/device/numa_node" 2>/dev/null)"
done > "$E/nvme.txt"
which nvcc && nvcc --version > "$E/nvcc.txt" 2>&1
(command -v fio && fio --version) > "$E/fio.txt" 2>&1

# GPU count and idleness
n_seen=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "gpus_visible: $n_seen" >> "$E/summary.txt"
[ "$NGPU" -le "$n_seen" ] || fail+=("GPUS lists $NGPU GPUs but only $n_seen visible")
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
echo "gpu_processes_running: $busy" >> "$E/summary.txt"
[ "$busy" -eq 0 ] || fail+=("other processes are using GPUs (see gpu_busy.csv)")

# DATA_DIR: exists, local block device, O_DIRECT works, free space
mkdir -p "$DATA_DIR" 2>/dev/null || fail+=("cannot create DATA_DIR=$DATA_DIR")
if [ -d "$DATA_DIR" ]; then
  findmnt -T "$DATA_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS > "$E/data_dir_mount.txt" 2>&1
  fstype=$(findmnt -T "$DATA_DIR" -no FSTYPE 2>/dev/null)
  echo "data_dir_fstype: $fstype" >> "$E/summary.txt"
  case "$fstype" in
    tmpfs|nfs*|lustre|cifs|fuse*|overlay) fail+=("DATA_DIR filesystem is $fstype; need local ext4/xfs on NVMe") ;;
  esac
  free_gib=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc 0-9)
  echo "data_dir_free_gib: $free_gib" >> "$E/summary.txt"
  [ "${free_gib:-0}" -ge 180 ] || fail+=("DATA_DIR has ${free_gib}G free, need >= 180G")
  if dd if=/dev/zero of="$DATA_DIR/.odirect_test" bs=1M count=64 oflag=direct status=none 2>/dev/null &&
     dd if="$DATA_DIR/.odirect_test" of=/dev/null bs=1M iflag=direct status=none 2>/dev/null; then
    echo "o_direct: ok" >> "$E/summary.txt"
  else
    fail+=("O_DIRECT read/write failed in DATA_DIR")
  fi
  rm -f "$DATA_DIR/.odirect_test"
fi

command -v nvcc >/dev/null || fail+=("nvcc not found (CUDA toolkit needed to build)")
command -v python3 >/dev/null || fail+=("python3 not found")
command -v fio >/dev/null || echo "warning: fio missing -> stage 10 and the write-interference stage are skipped" >> "$E/summary.txt"
command -v numactl >/dev/null || echo "warning: numactl missing -> no NUMA binding" >> "$E/summary.txt"

if [ -n "$VLLM_PY" ]; then
  "$VLLM_PY" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)" \
    > "$E/vllm.txt" 2>&1 || echo "warning: VLLM_PY import failed -> stages 30/40 skipped" >> "$E/summary.txt"
  grep -q "vllm 0.10.1.1" "$E/vllm.txt" || echo "warning: vllm is not 0.10.1.1 (connector targets 0.10.1.1)" >> "$E/summary.txt"
else
  echo "warning: VLLM_PY empty -> stages 30/40 skipped" >> "$E/summary.txt"
fi

printf '%s\n' "${fail[@]}" > "$E/failures.txt"
cat "$E/summary.txt"
if [ ${#fail[@]} -gt 0 ]; then
  printf 'FAIL: %s\n' "${fail[@]}"
  stage_end 00_preflight fail "$(IFS='; '; echo "${fail[*]}")"
  exit 1
fi
stage_end 00_preflight ok
