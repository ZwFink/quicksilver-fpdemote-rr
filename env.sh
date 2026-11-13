# Environment for every step (Tuolumne, MI300A). Source it from any directory.
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source /etc/profile.d/z00_lmod.sh
module load rocm/6.4.3 cmake/3.29.2 python/3.12.2 || return 1
export ROCM_PATH=/opt/rocm-6.4.3
LLVM=$ROCM_PATH/llvm
GPU_ARCH=gfx942
MNEME=$W/install/mneme
if [ -d "$W/venv" ]; then source "$W/venv/bin/activate"; fi
export PATH=$MNEME/bin:$PATH
export PYTHONPATH=$MNEME/python
export DEMOTE_PLUGIN=$W/build/fpdemote/libDemoteFP.so
# Mneme replays through Proteus's serial codegen path, which is the one that runs pass plugins.
export PROTEUS_CODEGEN=serial PROTEUS_USE_STORED_CACHE=0
# Compile each kernel once, without specializing on arguments or launch dimensions.
export PROTEUS_SPECIALIZE_ARGS=0 PROTEUS_SPECIALIZE_DIMS=0 PROTEUS_SPECIALIZE_DIMS_RANGE=0
QSARGS="-n 20000 -X 8 -Y 8 -Z 8 -x 8 -y 8 -z 8 -N 1"
