"""Lab 3 - Smart Semaphore Control. Single entry point.

    python main.py                 # run Parts A, B and C (~22 min)
    python main.py --quick         # smoke run (~1 min), writes to results/quick/ only
    python main.py --part A        # just one part (B and C reuse models/ from A)
    python main.py --fresh         # regenerate the GA training set instead of using the cache
    python main.py --data path/to/METR-LA.h5

Outputs: results/figures/*.png, results/tables/*.{csv,md,json}, models/*.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from semaphore import report as R  # noqa: E402
from semaphore.config import DEFAULT  # noqa: E402
from semaphore.fnn import FNN  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "kaggle" / "METR-LA.h5"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="short smoke run")
    ap.add_argument("--fresh", action="store_true", help="regenerate the GA training set")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="METR-LA .h5 (Kaggle) or cleaned .csv")
    args = ap.parse_args()

    from experiments import part_a, part_b, part_c

    if args.quick:
        R.use_quick_dirs()
    R.setup()
    cfg = DEFAULT
    t0 = time.time()
    net3 = net4 = None
    if args.part in ("A", "all"):
        print("=" * 70 + "\nPART A - isolated intersection\n" + "=" * 70)
        net3, _ = part_a.run(cfg, quick=args.quick, fresh=args.fresh)
    if args.part in ("B", "C", "all"):
        if net3 is None:
            p = R.MODELS / "fnn3_trained.npz"
            if not p.exists():
                sys.exit("models/fnn3_trained.npz missing - run Part A first")
            net3 = FNN.load(p, cfg.ga.offset_frac)
    if args.part in ("B", "all"):
        print("=" * 70 + "\nPART B - 3x3 network\n" + "=" * 70)
        net4, _ = part_b.run(cfg, net3, quick=args.quick)
    if args.part in ("C", "all"):
        print("=" * 70 + "\nPART C - METR-LA real sensor data\n" + "=" * 70)
        if net4 is None:
            net4 = net3.expand_to_4_inputs()
        if not Path(args.data).exists():
            sys.exit(f"{args.data} not found - download the Kaggle dataset (see README)")
        part_c.run(cfg, net3, net4, args.data, quick=args.quick)
    print(f"\nDone in {time.time() - t0:.0f} s. Figures: {R.FIG}  Tables: {R.TAB}")


if __name__ == "__main__":
    main()
