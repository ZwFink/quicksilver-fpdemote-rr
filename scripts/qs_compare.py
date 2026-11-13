"""Print one number: the relative L2 error of the Quicksilver scalar-flux tally after a
replay, against the recorded FP64 output.

    error = ||flux_replay - flux_recorded||_2 / ||flux_recorded||_2

The flux is collected over all domains and tasks. The script finds it in the memory images
by following pointers from the kernel's MonteCarlo* argument, with the struct offsets that
layout/layout.cc prints.

Usage:
  python qs_compare.py --record-db run-rec/db/X.json --layout layout/layout.json \
      --image replay/run0.npz
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mneme_snapshot import MemoryImage, load_image, read_snapshot  # noqa: E402


def qs_vector(img: MemoryImage, L: dict, addr: int):
    data = img.u64(addr + L["qs_vector<int>::_data"])
    size = img.i32(addr + L["qs_vector<int>::_size"])
    return data, size


def scalar_flux(img: MemoryImage, mc: int, L: dict) -> np.ndarray:
    tallies = img.u64(mc + L["MonteCarlo::_tallies"])
    domains, num_domains = qs_vector(img, L, tallies + L["Tallies::_scalarFluxDomain"])
    parts = []
    for d in range(num_domains):
        domain = domains + d * L["sizeof(ScalarFluxDomain)"]
        tasks, num_tasks = qs_vector(img, L, domain + L["ScalarFluxDomain::_task"])
        for t in range(num_tasks):
            storage = tasks + t * L["sizeof(ScalarFluxTask)"] + L["ScalarFluxTask::_scalarFluxCellStorage"]
            data = img.u64(storage + L["BulkStorage<double>::_bulkStorage"])
            size = img.i32(storage + L["BulkStorage<double>::_size"])
            parts.append(img.array(data, np.float64, size))
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record-db", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--image", required=True, help="memory image after the replay (.npz)")
    a = p.parse_args()

    L = json.load(open(a.layout))
    insts = json.load(open(a.record_db))["instances"]
    if len(insts) != 1:
        raise SystemExit(f"expected one recorded instance, found {len(insts)}")
    inst = next(iter(insts.values()))
    base = os.path.dirname(os.path.abspath(a.record_db))
    prologue = read_snapshot(os.path.join(base, inst["Prologue"]))
    recorded = read_snapshot(os.path.join(base, inst["Epilogue"]), base=prologue)
    replay = load_image(a.image)

    mc = struct.unpack("<Q", prologue.args[0])[0]
    ref = scalar_flux(recorded, mc, L)
    cand = scalar_flux(replay, mc, L)
    if ref.shape != cand.shape:
        raise SystemExit(f"scalar flux size differs: recorded {ref.size}, replay {cand.size}")
    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        raise SystemExit("recorded scalar flux is all zero")
    print(f"scalar flux relative L2 error: {np.linalg.norm(cand - ref) / ref_norm:.3e}")


if __name__ == "__main__":
    main()
