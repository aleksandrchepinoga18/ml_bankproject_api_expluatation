# monitoring/check_score_drift.py
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime



PSI_THRESHOLD = 0.1
KS_PVALUE_THRESHOLD = 0.05

def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)
    expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
    actual = (actual - actual.min()) / (actual.max() - actual.min() + 1e-8)
    
    bins_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    expected_bins = np.histogram(expected, bins=bins_edges)[0] + 1
    actual_bins = np.histogram(actual, bins=bins_edges)[0] + 1

    expected_dist = expected_bins / expected_bins.sum()
    actual_dist = actual_bins / actual_bins.sum()
    psi = np.sum((expected_dist - actual_dist) * np.log(expected_dist / actual_dist))
    return psi

def check_score_drift():
    # Референсные скоры
    ref_path = "monitoring/reference/reference_scores.parquet"
    if not os.path.exists(ref_path):
        print(" Нет референсных скоров — пропускаем проверку")
        return False

    ref_scores = pd.read_parquet(ref_path)["score"].dropna().values

    # Свежие скоры
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    scores = []
    for date in [today, yesterday]:
        log_file = f"monitoring/logs/predictions_{date}.jsonl"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    scores.append(entry["score"])
    
    if len(scores) < 10:
        print("  Недостаточно новых скоров для анализа")
        return False

    current_scores = np.array(scores)

    # PSI и KS
    psi = calculate_psi(ref_scores, current_scores)
    ks_stat, pval = stats.ks_2samp(ref_scores, current_scores)

    drift_detected = (psi > PSI_THRESHOLD) or (pval < KS_PVALUE_THRESHOLD)

    print(f" PSI по скорам: {psi:.4f}")
    print(f" KS p-value: {pval:.4f}")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": "score_drift",
        "psi": float(psi),
        "ks_pvalue": float(pval),
        "drift_detected": bool(drift_detected),
        "n_ref": len(ref_scores),
        "n_current": len(current_scores)
    }

    with open("monitoring/drift_logs/drift_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    if drift_detected:
        print(" Обнаружен дрейф скоров!")
    else:
        print(" Дрейф скоров не обнаружен")
    
    return drift_detected

if __name__ == "__main__":
    check_score_drift()