# Quicksilver: FP64 record, replay with FP32 demotion

This demo uses Mneme to record the Quicksilver `CycleTrackingKernel` in full FP64.
Then the demo uses the Mneme Python API to replay the recording with the
[DemoteFP](https://github.com/ZwFink/fpdemote) pass plugin.
This plugin computes each `double` operation in `float`.
Then the demo compares the output of the replay with the FP64 output of the recording.
The result is one number: the relative L2 error of the scalar flux.

The purpose of the demo is to show that you can use the recording of a kernel as a harness for experiments on precision.
You do not have to build the application again or run it again.

The demo is for Tuolumne (MI300A, gfx942) with ROCm 6.4.3.
The demo does not need anything outside this directory.

## Run the demo

```bash
./setup.sh     # clones and builds everything, about 20 minutes
./run-all.sh   # records, replays and compares
```

Both scripts load their environment from `env.sh` (modules `rocm/6.4.3`, `cmake/3.29.2`, `python/3.12.2`).
`run-all.sh` needs a GPU.
The login nodes on Tuolumne have a GPU.
If you do not use a login node, run `run-all.sh` inside `flux alloc -N1 --bank=<your bank>`.

To work interactively, run `source env.sh`.
You can do this from any directory.

`setup.sh` clones these repositories into `src/`:

| Repository | Branch | Commit at the date of this README |
|---|---|---|
| [LLNL/Mneme](https://github.com/LLNL/Mneme) | `develop` | 6dfd0bb (#243, pass-plugin registration) |
| [LLNL/Quicksilver](https://github.com/LLNL/Quicksilver) | `master` | eb68bb8 (unmodified) |
| [ZwFink/fpdemote](https://github.com/ZwFink/fpdemote) | `main` | ec1ae16 |

Then `setup.sh` creates a Python venv (`numpy`, `PyYAML`, `optuna`, `scipy`) in `venv/`.
Then it builds these items:

- The DemoteFP plugin, into `build/fpdemote/libDemoteFP.so`.
  The build uses the LLVM of ROCm, which is also the LLVM that Proteus uses.
- Mneme, into `install/mneme`.
  The build also includes the Proteus version that Mneme pins (`src/mneme/PROTEUS_VERSION`).
  CMake fetches this version of Proteus.
- Quicksilver, into `qs/qs`.
  `build-qs.sh` builds Quicksilver for HIP, without MPI, and with `-fgpu-rdc`.
  It compiles Quicksilver with the Proteus pass and links it with Proteus and Mneme.
- `layout/layout.json`.
  This file holds the struct offsets of the QS types.
  The comparison uses these offsets to find the scalar flux in device memory.

You can run `setup.sh` again at any time.
It does not clone a repository again if the clone is already in `src/`.
It does not create the venv again if the venv exists.
It builds all other items again.

To get new commits, do `git pull` in each `src/*` directory.
Then run `setup.sh` again.

Do not move this directory after the build.
The venv and the run paths of the libraries are absolute.
To share or move the demo, copy only the scripts (see [Contents](#contents)) to the new location.
Then run `setup.sh` in the new location.

## What `run-all.sh` does

1. **Record.**
   `run-all.sh` runs `mneme record -rdb db -mr 1 -- ../qs/qs -n 20000 -X 8 -Y 8 -Z 8 -x 8 -y 8 -z 8 -N 1` in `run-rec/`.
   This command records the first `CycleTrackingKernel` launch of cycle 0 (grid 16, block 128, 2000 particles), in FP64.
2. **Replay.**
   `scripts/qs_replay.py` does these steps:
   - It loads the recording with the Mneme `BaseExecutor`.
   - It registers the plugin.
   - It maps the memory of the recording at the same device addresses as in the recording.
   - It JIT-compiles the IR of the recording with the pipeline `demote-fp,default<O3>`.
   - It launches the kernel one time.
   - It saves the device memory after the launch to `replay/run0.npz`.

   The `replay/` directory also holds the JIT output (`kernel.opt.ll`, `kernel.s`, `kernel.o`).
   It also holds `summary.json`, which contains the pipeline, the register count and the counts of FP operations in the ISA.
3. **Compare.**
   `scripts/qs_compare.py` compares the output of the replay with the FP64 output of the recording.
   The script prints the relative L2 error of the scalar flux.

### The demotion

The `demote-fp` pass demotes the following categories of FP64 operations to FP32:

- **arithmetic:** `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg` and `fcmp`, and the `sqrt`, `fabs`, `fma` and `fmuladd` intrinsics.
- **math:** The plugin changes calls to the AMD math library (`__ocml_*_f64`) and the equivalent `llvm.*.f64` intrinsics to their `f32` versions.
- **atomics:** The plugin changes `atomicrmw fadd`/`fsub` on `double` to a compare-and-swap loop that adds in `float`.
  QS uses these atomics to accumulate its tallies.

Loads, stores and the memory layout stay `double`.
The plugin converts each value to `float` for the operation, and then converts the result back to `double`.
The plugin runs before the O3 pipeline of Proteus.

The replay must use the serial code generation of Proteus (`PROTEUS_CODEGEN=serial` in `env.sh`).
This is the path that runs pass plugins.

### The comparison

For simplicity, `qs_compare.py` compares the scalar-flux tally, which is the main result of QS.
The script follows the pointers from the `MonteCarlo` kernel argument to find the scalar flux of all domains.
Then it calculates the relative L2 error:

```
error = ||flux_replay - flux_recorded||_2 / ||flux_recorded||_2
```

This number is a simple summary of the difference.

## Mneme Python API gaps

Mneme works but doesn't play perfectly with this replay pattern out of the box.
The following pain points can be smoothed over with some infrastructure work,
but hopefully what we have here is enough to get off the ground.

- The API cannot read device memory after a launch.
  Thus, the script copies each buffer of the recording back with `hipMemcpy`.
  It calls `hipMemcpy` through `ctypes` on `libamdhip64`.
- `MemStateRef` is opaque.
  `MemBufferRef` does not give access to its bytes.
- The API has no single public call that builds and launches a recording with a custom pipeline.
  Thus, the script itself does the same build steps as `BaseExecutor` (`jit.optimize`, `jit.codegen_object`).

## Contents

| Path | What it is |
|---|---|
| `env.sh` | The modules, the paths, the settings for Proteus, and the QS arguments (`QSARGS`). |
| `setup.sh` | Clones the sources, creates the venv, and runs the build scripts. |
| `build-mneme.sh`, `build-qs.sh`, `layout/build.sh` | The scripts for the individual builds. |
| `run-all.sh` | Records, replays and compares. |
| `scripts/qs_replay.py` | Replays one instance from the recording with a pipeline and pass plugins. Saves the memory after the launch. |
| `scripts/qs_compare.py` | Finds the scalar flux in the memory images and prints its relative L2 error. |
| `scripts/mneme_snapshot.py` | A reader for Mneme snapshot files, in pure Python. Both scripts use it. |
| `layout/layout.cc` | Prints the QS struct offsets that `qs_compare.py` uses, as `layout.json`. |

The scripts generate these items: `src/`, `venv/`, `build/`, `install/`, `qs/`, `run-rec/`, `replay/`, `compare.txt`, and the logs.
