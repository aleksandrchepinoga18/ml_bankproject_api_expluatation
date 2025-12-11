from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monitoring.log_predictions import log_prediction

app = Flask(__name__)

# Загрузка модели
MODEL_PATH = "models/lightgbm_model.pkl"
THRESHOLD_PATH = "models/lightgbm_best_threshold.pkl"

model = joblib.load(MODEL_PATH)
best_threshold = joblib.load(THRESHOLD_PATH)
feature_names = model.feature_name_

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not isinstance(data, list):
            data = [data]

        X = pd.DataFrame(data)[feature_names]
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= best_threshold).astype(int)

        # ЛОГИРУЕМ КАЖДЫЙ СКОР
        for i in range(len(X)):
            log_prediction(
                features=X.iloc[i].to_dict(),
                score=float(proba[i])
            )

        result = [
            {"prediction": int(p), "risk_probability": float(pr)}
            for p, pr in zip(pred, proba)
        ]
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)