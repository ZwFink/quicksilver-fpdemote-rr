"""Replay one recorded Mneme kernel instance under a chosen LLVM pipeline and keep
the raw post-launch device memory, instead of relying on Mneme's exact verify.

Uses only Mneme's public Python pieces plus existing native exports:
  * BaseExecutor: loads the recording, registers pass plugins, maps the prologue
    at its recorded device addresses (open()), and links the recorded IR.
  * jit.optimize / jit.codegen_object: the same JIT steps BaseExecutor._build runs.
  * DeviceModule / DeviceFunction.profile(repeats=1): restores the prologue and
    launches once.
  * hipMemcpy (ctypes on libamdhip64): reads every recorded blob back to the host,
    since Mneme's Python API has no way to read device memory after a launch.

Each repeat writes <out>/run<i>.npz with the post-launch image (same layout as the
recorded snapshots, see mneme_snapshot.save_image). Mneme's own verify verdict is
recorded in <out>/summary.json for reference only.

Example:
  python qs_replay.py --record-db run-rec/db/X.json --out replay \
      --pass-plugin libDemoteFP.so --pipeline 'demote-fp,default<O3>'
  python qs_replay.py --record-db run-rec/db/X.json --out replay-fp64 \
      --pipeline 'default<O3>' --repeats 3
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mneme_snapshot import MemoryImage, Blob, read_snapshot, save_image  # noqa: E402

from mneme.device import DeviceModule  # noqa: E402
from mneme.llvm import ffi  # noqa: E402
from mneme.proteus import jit  # noqa: E402
from mneme.replay_executor import BaseExecutor  # noqa: E402

HIP_MEMCPY_DEVICE_TO_HOST = 2
LLVM_OBJDUMP = os.path.join(os.environ["ROCM_PATH"], "llvm", "bin", "llvm-objdump")


def _hip():
    hip = ctypes.CDLL("libamdhip64.so.6", mode=ctypes.RTLD_GLOBAL)
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemcpy.restype = ctypes.c_int
    hip.hipDeviceSynchronize.restype = ctypes.c_int
    return hip


def read_device_image(hip, layout: MemoryImage) -> MemoryImage:
    """Copy every blob of the recorded layout off the device."""
    err = hip.hipDeviceSynchronize()
    if err:
        raise RuntimeError(f"hipDeviceSynchronize failed: {err}")
    out = MemoryImage(args=list(layout.args))
    for addr, blob in layout.blobs.items():
        host = np.empty(len(blob.data), dtype=np.uint8)
        err = hip.hipMemcpy(host.ctypes.data, addr, len(host), HIP_MEMCPY_DEVICE_TO_HOST)
        if err:
            raise RuntimeError(f"hipMemcpy D2H of blob {addr:#x} ({len(host)} B) failed: {err}")
        out.blobs[addr] = Blob(addr, blob.actual_size, host, blob.metadata)
    if layout.globals:
        raise NotImplementedError(
            "the recording has device globals; reading them back needs the loaded "
            "module's global addresses, which this script does not resolve"
        )
    return out


def object_bytes(mem_buffer) -> bytes:
    lib = ffi.lib
    lib.LLVMPY_GetBufferStart.argtypes = [ffi.LLVMMemBufferRef]
    lib.LLVMPY_GetBufferStart.restype = ctypes.c_void_p
    start = lib.LLVMPY_GetBufferStart(mem_buffer)
    return ctypes.string_at(start, mem_buffer.get_size())


def isa_op_counts(asm: str) -> dict:
    """Count VALU floating-point instructions by precision suffix in a disassembly."""
    counts = {}
    for line in asm.splitlines():
        parts = line.split()
        if not parts:
            continue
        op = parts[0]
        if not op.startswith("v_"):
            continue
        for suffix in ("_f64", "_f32", "_f16"):
            if suffix in op:
                counts[suffix[1:]] = counts.get(suffix[1:], 0) + 1
                counts.setdefault("ops", {})
                counts["ops"][op] = counts["ops"].get(op, 0) + 1
                break
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record-db", required=True, help="recording JSON (one kernel)")
    p.add_argument("--record-id", default=None, help="instance id; inferred if the JSON has one")
    p.add_argument("--pipeline", required=True, help="LLVM opt -passes textual pipeline")
    p.add_argument("--pass-plugin", action="append", default=[], help="pass plugin .so (repeatable)")
    p.add_argument("--purge-autoinit", action="store_true",
                   help="run Mneme's purge-autoinit pipeline first (as tracked builds do)")
    p.add_argument("--codegen-opt", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1, help="launches, each from the restored prologue")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hip = _hip()

    ex = BaseExecutor(record_db=a.record_db, record_id=a.record_id, iterations=1, warmup=0,
                      device_id=a.device, pass_plugins=a.pass_plugin)
    kd = ex.kernel_descr
    layout = read_snapshot(kd.prologue.fn)
    summary = {
        "record_db": os.path.abspath(a.record_db), "record_id": ex.record_id,
        "kernel": kd.kernel_name, "grid": kd.grid_dim.to_dict(), "block": kd.block_dim.to_dict(),
        "shared_mem": kd.shared_mem, "pipeline": a.pipeline, "pass_plugins": a.pass_plugin,
        "purge_autoinit": a.purge_autoinit, "codegen_opt": a.codegen_opt,
        "device_arch": ex.device_arch, "proteus_codegen": os.environ.get("PROTEUS_CODEGEN"),
        "runs": [],
    }

    root_ir = ex.link_ir()
    # Raises jit.InvalidPassPipelineError (e.g. a misspelled plugin pass) before any
    # device state is mapped.
    jit.check_pass_pipeline(root_ir, ex.device_arch, a.pipeline, a.codegen_opt)
    with ex:
        summary["noop_verifies"] = bool(ex.noop_verifies)
        mod = root_ir.clone()
        t0 = time.time()
        if a.purge_autoinit:
            jit.optimize(mod, ex.device_arch, "purge-autoinit", a.codegen_opt)
        jit.optimize(mod, ex.device_arch, a.pipeline, a.codegen_opt)
        summary["opt_seconds"] = time.time() - t0
        (out / "kernel.opt.ll").write_text(str(mod))
        mem_buffer = jit.codegen_object(mod, ex.device_arch, a.codegen_opt)
        (out / "kernel.o").write_bytes(object_bytes(mem_buffer))
        asm = subprocess.run([LLVM_OBJDUMP, "-d", "--mcpu=" + ex.device_arch.split(":")[0],
                              str(out / "kernel.o")], check=True, capture_output=True, text=True).stdout
        (out / "kernel.s").write_text(asm)
        summary["isa_fp_ops"] = isa_op_counts(asm)

        with DeviceModule.from_MemBuffer(mem_buffer) as dm:
            fn = dm.get_function(kd.kernel_name)
            summary["reg_usage"] = fn.reg_usage
            summary["local_mem"] = fn.local_mem
            for i in range(a.repeats):
                fn.profile(kd.grid_dim, kd.block_dim, ex.prologue._state, ex.epilogue._state, kd.shared_mem, 1)
                img = read_device_image(hip, layout)
                save_image(str(out / f"run{i}.npz"), img)
                mneme_verified = bool(ex.prologue == ex.epilogue)
                summary["runs"].append({"image": f"run{i}.npz", "mneme_verify": mneme_verified})
                print(f"[qs_replay] run {i}: image saved, Mneme exact verify = {mneme_verified}", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
