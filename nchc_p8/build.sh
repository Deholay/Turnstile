#!/bin/bash
# Build on the nano5 login node: module load cuda/12.6. Same flags as bundle_8gpu/scripts/01_build.sh
# on H200 (compute capability 9.0).
set -e
cd "$(dirname "$0")"
ARCH="-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
nvcc -O3 -std=c++17 $ARCH -Xcompiler -pthread -o restore src/restore.cu
echo built ./restore
