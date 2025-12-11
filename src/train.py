import lightgbm as lgb
import numpy as np
import joblib
import os
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb
import joblib
import os

def train_lightgbm(X_train, y_train, random_state=42):
    """
    Обучение LightGBM с фиксированными гиперпараметрами из Colab.
    """
    os.makedirs('models', exist_ok=True)

    # 🔑 Фиксированные гиперпараметры из Colab (лучший результат)
    best_params = {
        'colsample_bytree': 0.6705331755251293,
        'learning_rate': 0.04404205637217672,
        'max_depth': 9,
        'min_child_samples': 40,
        'n_estimators': 347,
        'num_leaves': 118,
        'reg_alpha': 0.22855002179729966,
        'reg_lambda': 0.17495492709593619,
        'subsample': 0.9910841716647179
    }

    # Базовые параметры
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': -1
    }

    # Объединяем
    final_params = {**base_params, **best_params}

    # Создаём и обучаем модель
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train, y_train)

    # Сохраняем
    joblib.dump(model, 'models/lightgbm_model.pkl')

    return model, best_params, X_train.columns.tolist()