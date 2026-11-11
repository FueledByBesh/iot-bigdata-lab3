"""Standalone METR-LA cleaning script (Part C deliverable).

    python scripts/clean_metrla.py [--in data/kaggle/METR-LA.h5] [--out data/processed]

Steps (see semaphore/metrla.py for the exact rules):
  1. decode the pandas-0.15 HDF5 file with h5py (modern pandas cannot read it)
  2. 0 mph -> missing
  3. gaps <= 30 min: time-linear interpolation
  4. longer gaps (incl. ~7.5 days of network-wide outages): the sensor's
     median for the same 5-min slot of the week
  5. clip to [1, 70] mph
Writes the cleaned speeds, a 0/1 mask of originally missing readings, and a
JSON profile/cleaning report.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semaphore import metrla as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "kaggle" / "METR-LA.h5"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = M.load_metrla(a.inp)
    prof = M.profile(raw)
    clean, missing, rep = M.clean(raw)
    clean.round(3).to_csv(out / "metrla_speed_clean.csv.gz")
    missing.astype("int8").to_csv(out / "metrla_missing_mask.csv.gz")
    (out / "metrla_cleaning_report.json").write_text(json.dumps({**prof, **rep}, indent=2))
    print(json.dumps({**prof, **rep}, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
