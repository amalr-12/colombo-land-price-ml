"""Step 3: quality gate. Promote the candidate model only if it is good enough.

Rules (config/params.json -> quality_gate):
  * test R2 (log) must be at least `min_test_r2_log`, and
  * it must not be worse than the currently deployed model by more than `max_r2_drop_vs_current`.
If both pass, the deployed model is backed up and replaced by the candidate.
"""
import pickle
import shutil
from pathlib import Path


def check(candidate_metrics, current_path, gate):
    new_r2 = candidate_metrics["Test R2 (log)"]
    reasons = []
    if new_r2 < gate["min_test_r2_log"]:
        reasons.append(f"test R2 {new_r2:.4f} is below the minimum {gate['min_test_r2_log']}")

    cov = candidate_metrics.get("NGBoost", {}).get("80% range coverage")
    if cov is not None and not gate["ngboost_coverage80_min"] <= cov <= gate["ngboost_coverage80_max"]:
        reasons.append(f"NGBoost 80% ranges cover {cov:.1%} of test prices, outside "
                       f"{gate['ngboost_coverage80_min']:.0%}-{gate['ngboost_coverage80_max']:.0%}")

    current_r2 = None
    if Path(current_path).exists():
        with open(current_path, "rb") as f:
            current_r2 = pickle.load(f)["test_metrics"]["Test R2 (log)"]
        if new_r2 < current_r2 - gate["max_r2_drop_vs_current"]:
            reasons.append(f"test R2 {new_r2:.4f} is more than {gate['max_r2_drop_vs_current']} below "
                           f"the deployed model's {current_r2:.4f}")
    return not reasons, reasons, current_r2


def promote(candidate_path, current_path, candidate_metrics, gate):
    ok, reasons, current_r2 = check(candidate_metrics, current_path, gate)
    cur = f"{current_r2:.4f}" if current_r2 is not None else "none"
    print(f"quality gate: candidate R2 {candidate_metrics['Test R2 (log)']:.4f} vs deployed {cur}")
    if not ok:
        print("REJECTED - deployed model left unchanged:")
        for r in reasons:
            print("  -", r)
        return False
    current_path = Path(current_path)
    if current_path.exists():
        backup = current_path.with_name(current_path.stem + ".prev.pkl")
        shutil.copy2(current_path, backup)
        print(f"backed up deployed model to {backup.name}")
    shutil.copy2(candidate_path, current_path)
    ngb_candidate = Path(candidate_path).with_name("ngboost_model.pkl")
    if ngb_candidate.exists():
        shutil.copy2(ngb_candidate, current_path.with_name("ngboost_model.pkl"))
    print(f"PROMOTED - {current_path.name} (and ngboost_model.pkl) replaced by the candidates")
    return True
