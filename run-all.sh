#!/bin/bash
# Records QS in FP64 (1 cycle, first CycleTrackingKernel launch), replays the recording through
# the Mneme Python API with the DemoteFP plugin, and compares the replay with the recorded
# FP64 output (one number: the relative L2 error of the scalar flux). Needs a GPU; run setup.sh first.
set -eo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$W/env.sh"
set -u
cd "$W"

rm -rf run-rec replay && mkdir -p run-rec
(cd run-rec && mneme record -rdb db -mr 1 -- ../qs/qs $QSARGS > rec.log 2>&1)
DB=$(grep -l CycleTrackingKernel run-rec/db/*.json)

python scripts/qs_replay.py --record-db "$DB" --out replay --repeats 1 \
  --pass-plugin "$DEMOTE_PLUGIN" --pipeline 'demote-fp,default<O3>' > replay.log 2>&1
python scripts/qs_compare.py --record-db "$DB" --layout layout/layout.json \
  --image replay/run0.npz | tee compare.txt
