# monitoring/check_data_drift.py
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
from datetime import datetime

PSI_THRESHOLD = 0.2
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

def check_data_drift():
    # Загружаем референс
    ref_path = "monitoring/reference/reference_features.parquet"
    if not os.path.exists(ref_path):
        print("⚠️ Нет референсных фичей — пропускаем проверку дрейфа")
        return False

    ref_df = pd.read_parquet(ref_path)

    # Загружаем свежие логи (за вчера или сегодня)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    current_df = pd.DataFrame()
    for date in [today, yesterday]:
        log_file = f"monitoring/logs/predictions_{date}.jsonl"
        if not os.path.exists(log_file):
            continue
            
        features_list = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # пропускаем пустые строки
                    continue
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict) and entry["features"] is not None and isinstance(entry["features"], dict):
                        features_list.append(entry["features"])
                    else:
                        print(f"⚠️ Неверный формат в строке {line_num} файла {log_file}")
                except Exception as e:
                    print(f"❌ Ошибка в строке {line_num} файла {log_file}: {e}")
                    continue

        if features_list:
            df_features = pd.DataFrame(features_list)
            current_df = pd.concat([current_df, df_features], ignore_index=True)

    if current_df.empty:
        print("ℹ️ Нет новых данных для анализа дрейфа фичей")
        return False

    # Приводим к одному набору колонок
    common_features = ref_df.columns.intersection(current_df.columns)
    if len(common_features) == 0:
        print("⚠️ Нет общих фичей между референсом и текущими данными")
        return False

    ref_df = ref_df[common_features]
    current_df = current_df[common_features]

    # Проверяем каждую фичу
    drift_detected = False
    results = {}

    for col in common_features:
        ref_vals = ref_df[col].dropna()
        curr_vals = current_df[col].dropna()
        if len(ref_vals) < 10 or len(curr_vals) < 10:
            continue

        # KS-тест
        _, pval = stats.ks_2samp(ref_vals, curr_vals)
        
        # PSI
        psi = calculate_psi(ref_vals, curr_vals)

        results[col] = {"psi": float(psi), "ks_pvalue": float(pval)}
        
        if psi > PSI_THRESHOLD or pval < KS_PVALUE_THRESHOLD:
            print(f"🚨 Дрейф в фиче '{col}': PSI={psi:.4f}, KS p-value={pval:.4f}")
            drift_detected = True

    # Создаём папку для логов, если её нет
    os.makedirs("monitoring/drift_logs", exist_ok=True)
    
    # Логируем
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": "feature_drift",
        "drift_detected": bool(drift_detected),
        "details": results
    }
    with open("monitoring/drift_logs/drift_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if not drift_detected:
        print("✅ Дрейф фичей не обнаружен")
    return drift_detected

if __name__ == "__main__":
    check_data_drift()