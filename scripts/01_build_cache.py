"""Build the order-spectrum cache from data/.  Runs once; ~30-60 min."""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drivesentinel.features import build_cache

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=None, help="first N recordings (smoke test)")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also cache the KB combined-class bearings")
    a = ap.parse_args()
    build_cache(n_jobs=a.jobs, limit=a.limit, include_excluded=a.include_excluded)
