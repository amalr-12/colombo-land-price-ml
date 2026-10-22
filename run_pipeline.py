"""Retraining pipeline: raw data -> clean data -> trained candidate -> quality gate.

    python run_pipeline.py              # prepare + train, write the candidate to artifacts/
    python run_pipeline.py --promote    # ... then replace land_price_model.pkl if the candidate passes the gate

Outputs in artifacts/: land_prices_clean.csv, land_price_model.pkl (candidate), metrics.json.
"""
import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.prepare import prepare  # noqa: E402
from src.promote import promote  # noqa: E402
from src.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=ROOT / "Land prices of Colombo district-Sri Lanka.xlsx", type=Path)
    ap.add_argument("--params", default=ROOT / "config" / "params.json", type=Path)
    ap.add_argument("--out", default=ROOT / "artifacts", type=Path)
    ap.add_argument("--promote", action="store_true", help="replace the deployed model if the gate passes")
    args = ap.parse_args()

    params = json.loads(args.params.read_text())
    args.out.mkdir(exist_ok=True)
    total = time.time()

    t = time.time()
    df, prep_report = prepare(args.raw, params)
    df.to_csv(args.out / "land_prices_clean.csv", index=False)
    print(f"[1/3] prepare: {prep_report}  ({time.time() - t:.0f}s)")

    t = time.time()
    df = df.copy()
    bundle, metrics = train(df, params)
    candidate = args.out / "land_price_model.pkl"
    with open(candidate, "wb") as f:
        pickle.dump(bundle, f)
    metrics["data"] = prep_report
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[2/3] train: R2 (log) {metrics['Test R2 (log)']:.4f} | median APE {metrics['Test median APE']:.1%} | "
          f"band accuracy {metrics['Band accuracy']:.1%}  ({time.time() - t:.0f}s)")

    if args.promote:
        ok = promote(candidate, ROOT / "land_price_model.pkl", metrics, params["quality_gate"])
        print(f"[3/3] gate: {'passed' if ok else 'failed'}")
        if not ok:
            sys.exit(1)
    else:
        print("[3/3] gate: skipped (run with --promote to apply it)")
    print(f"done in {(time.time() - total) / 60:.1f} min")


if __name__ == "__main__":
    main()
