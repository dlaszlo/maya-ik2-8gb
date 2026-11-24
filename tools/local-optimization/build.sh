#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda_dir="${IK2_CUDA_ROOT:-/home/dlaszlo/local-ai/cuda-13.0}"
cmake -S "$repo_dir" -B "$repo_dir/build-opt" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER="$cuda_dir/bin/nvcc" -DCUDAToolkit_ROOT="$cuda_dir"
cmake --build "$repo_dir/build-opt" --target maya-ik2 -j 4
