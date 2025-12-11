import joblib
import pandas as pd

def load_model(path="models/lightgbm_model.pkl"):
    return joblib.load(path)

def predict(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]

