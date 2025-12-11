# monitoring/simulate_labels.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

def simulate_labels(days_back=7):  # ← уменьшите до 7 дней для надёжности
    """Имитирует true_label для демонстрации (в реальности — из БД)"""
    all_features = []
    all_scores = []
    
    for i in range(days_back):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        log_file = f"monitoring/logs/predictions_{date}.jsonl"
        if not os.path.exists(log_file):
            continue
            
        with open(log_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if (isinstance(entry, dict) and 
                        "features" in entry and 
                        isinstance(entry["features"], dict) and
                        "score" in entry):
                        all_features.append(entry["features"])
                        all_scores.append(entry["score"])
                except Exception as e:
                    print(f"⚠️ Ошибка в {log_file}, строка {line_num}: {e}")
                    continue

    if not all_features:
        print("⚠️ Нет валидных логов для симуляции лейблов")
        return

    # Создаём датафрейм только из фичей
    df_features = pd.DataFrame(all_features)
    df = pd.DataFrame({"score": all_scores})
    df = pd.concat([df, df_features], axis=1)

    # Имитация лейблов
    np.random.seed(42)
    df["true_label"] = np.random.binomial(1, 1 - df["score"])  # дефолт = 1

    # Сохраняем
    output_file = "monitoring/logs/predictions_with_labels.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Симулировано {len(df)} лейблов и сохранено в {output_file}")

if __name__ == "__main__":
    simulate_labels()