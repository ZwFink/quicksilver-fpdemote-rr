#!/bin/bash
# Builds Mneme with Python bindings into install/mneme. Mneme's CMake fetches and builds
# Proteus at the commit in src/mneme/PROTEUS_VERSION and installs it into the same prefix.
set -eo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$W/env.sh"
set -u
cmake -S "$W/src/mneme" -B "$W/build/mneme" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$MNEME \
  -DLLVM_INSTALL_DIR=$LLVM -DCMAKE_C_COMPILER=$LLVM/bin/amdclang -DCMAKE_CXX_COMPILER=$LLVM/bin/amdclang++ \
  -DMNEME_GPU_BACKEND=hip -DMNEME_ENABLE_PYTHON=On -DMNEME_ENABLE_LOGGER=On \
  -DPython3_EXECUTABLE="$W/venv/bin/python" -DMNEME_PYTHON_SITE_PACKAGES=$MNEME/python
cmake --build "$W/build/mneme" -j64
cmake --install "$W/build/mneme"
