#!/bin/bash
# Builds upstream Quicksilver (HIP, no MPI, -fgpu-rdc) into qs/qs with the Proteus and Mneme
# compiler flags (what Mneme's add_mneme CMake helper adds), against install/mneme.
set -eo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$W/env.sh"
set -u
SRC=$W/src/quicksilver/src
OUT=$W/qs
mkdir -p $OUT/obj
echo "#define GIT_HASH \"$(git -C $SRC rev-parse --short HEAD)\"" > $OUT/git_hash.hh
echo '#define GIT_VERS "upstream"' > $OUT/git_vers.hh
CXXFLAGS="-O3 -g -std=c++17 -DHAVE_HIP -x hip --offload-arch=$GPU_ARCH -fgpu-rdc -Wno-unused-result -I$OUT \
  -fplugin=$MNEME/lib64/libProteusPass.so -fpass-plugin=$MNEME/lib64/libProteusPass.so \
  -Xclang -mllvm -Xclang -force-proteus-jit-annotate-all -fplugin=$MNEME/lib64/libProteusASTChecker.so \
  -fno-discard-value-names -ftrivial-auto-var-init=zero -gline-tables-only -I$MNEME/include"
LDFLAGS="-O3 -fgpu-rdc --hip-link --offload-arch=$GPU_ARCH \
  -Xoffload-linker --load-pass-plugin=$MNEME/lib64/libProteusPassOffload.so \
  -L$MNEME/lib64 -lproteus -lmnemert -Wl,-rpath,$MNEME/lib64"
cd $SRC
pids=()
for f in *.cc; do
  # PhysicalConstants.hh already defines these constants in GPU builds.
  [ "$f" = PhysicalConstants.cc ] && continue
  $LLVM/bin/clang++ $CXXFLAGS -c $f -o $OUT/obj/${f%.cc}.o &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
$LLVM/bin/clang++ $LDFLAGS $OUT/obj/*.o -o $OUT/qs -L$ROCM_PATH/lib -lamdhip64 -Wl,-rpath,$ROCM_PATH/lib
echo "built $OUT/qs"
