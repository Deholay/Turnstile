#!/bin/bash
# Stage 01: build build/restore and build/libkvr.so for the GPUs in this machine.
source "$(dirname "$0")/lib.sh"
stage_begin 01_build
mkdir -p "$BIN"
cc=$(nvidia-smi -i "${GPU_ARR[0]}" --query-gpu=compute_cap --format=csv,noheader | tr -d ' .')
ARCH="-gencode arch=compute_${cc},code=sm_${cc} -gencode arch=compute_${cc},code=compute_${cc}"
echo "compute capability $cc" | tee "$SDIR/build.log"
set -o pipefail
nvcc -O3 -std=c++17 $ARCH -Xcompiler -pthread -o "$BIN/restore" "$B/src/restore.cu" 2>&1 | tee -a "$SDIR/build.log" &&
nvcc -O3 -std=c++17 $ARCH -Xcompiler -pthread,-fPIC -shared -o "$BIN/libkvr.so" "$B/src/kvr_lib.cu" 2>&1 | tee -a "$SDIR/build.log"
rc=$?
sha256sum "$B"/src/* "$BIN"/* > "$SDIR/sha256.txt" 2>/dev/null
[ $rc -eq 0 ] && stage_end 01_build ok || { stage_end 01_build fail "nvcc failed, see build.log"; exit 1; }
