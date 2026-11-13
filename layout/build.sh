#!/bin/bash
# Builds the QS struct-layout probe with Quicksilver's HIP flags and writes layout/layout.json.
set -eo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$W/env.sh"
set -u
cd "$W/layout"
$LLVM/bin/clang++ -O0 -std=c++17 -DHAVE_HIP -x hip --offload-arch=$GPU_ARCH -fgpu-rdc -Wno-unused-result -w \
  -I"$W/qs" -I"$W/src/quicksilver/src" layout.cc -c -o layout.o
$LLVM/bin/clang++ -fgpu-rdc --hip-link --offload-arch=$GPU_ARCH layout.o -o layout \
  -L$ROCM_PATH/lib -lamdhip64 -Wl,-rpath,$ROCM_PATH/lib
./layout > layout.json
