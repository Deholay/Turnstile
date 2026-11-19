#!/bin/bash
# Build on nano5 login node: module load cuda/12.6. sm_90 SASS for H200 + compute_90 PTX (JIT on RTX 5090).
set -e
cd "$(dirname "$0")"
ARCH="-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"
nvcc -O3 -std=c++17 $ARCH -Xcompiler -pthread -o restore restore.cu
nvcc -O3 -std=c++17 $ARCH -Xcompiler -pthread,-fPIC -shared -o libkvr.so kvr_lib.cu
echo built ./restore ./libkvr.so
