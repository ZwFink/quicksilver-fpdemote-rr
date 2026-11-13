#!/bin/bash
# Clones Mneme (develop), Quicksilver (master) and fpdemote (main), creates the Python venv,
# and builds the DemoteFP plugin, Mneme (with the Proteus it fetches), Quicksilver, and the
# layout probe. Takes about 20 minutes.
set -eo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$W/env.sh"
set -u

mkdir -p "$W/src"
clone() {  # clone <repo> <branch> <dir>
  [ -d "$3/.git" ] || git clone --quiet --branch "$2" "$1" "$3"
}
clone https://github.com/LLNL/Mneme.git develop "$W/src/mneme"
clone https://github.com/LLNL/Quicksilver.git master "$W/src/quicksilver"
clone https://github.com/ZwFink/fpdemote.git main "$W/src/fpdemote"

if [ ! -d "$W/venv" ]; then
  python3 -m venv "$W/venv"
  "$W/venv/bin/pip" install --quiet --upgrade pip
  "$W/venv/bin/pip" install --quiet numpy "PyYAML>=6.0" "optuna>=4.4" scipy
  source "$W/venv/bin/activate"
fi

# DemoteFP must build against the LLVM that Proteus uses, which is ROCm's.
cmake -S "$W/src/fpdemote" -B "$W/build/fpdemote" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$LLVM/bin/clang++ -DLLVM_DIR=$LLVM/lib/cmake/llvm
cmake --build "$W/build/fpdemote" -j16

"$W/build-mneme.sh"
"$W/build-qs.sh"
"$W/layout/build.sh"
echo "setup done"
