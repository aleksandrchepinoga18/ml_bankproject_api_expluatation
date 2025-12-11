# monitoring/check_model_quality.py
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score
import json
from datetime import datetime, timedelta
import json

# Загружаем лучший порог
import joblib
best_threshold = joblib.load("models/lightgbm_best_threshold.pkl")

def check_model_quality():
    # Ищем файл с предсказаниями и лейблами
    label_file = "monitoring/logs/predictions_with_labels.csv"
    if not os.path.exists(label_file):
        print("ℹ️ Нет файла с лейблами — пропускаем оценку качества")
        return None

    df = pd.read_csv(label_file)
    required_cols = {"score", "true_label"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ Нет колонок {required_cols} в {label_file}")
        return None

    df = df.dropna(subset=["score", "true_label"])
    if len(df) < 10:  #  можно поставить меньше для испытаний и проверок 
        print("ℹ️ Недостаточно данных с лейблами")
        return None

    y_true = df["true_label"].astype(int)
    y_pred_proba = df["score"]
    #y_pred = (y_pred_proba >= 0.5).astype(int)  # или используйте ваш порог
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)

    print(f"🎯 ROC-AUC: {auc:.4f}")
    print(f"🎯 F1-score: {f1:.4f}")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "component": "model_quality",
        "roc_auc": float(auc),
        "f1_score": float(f1),
        "n_samples": len(df)
    }

    with open("monitoring/drift_logs/drift_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return auc, f1

if __name__ == "__main__":
    check_model_quality()