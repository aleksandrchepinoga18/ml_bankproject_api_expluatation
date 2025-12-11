from src.data_preparation import *
from src.eda import run_eda
from src.train import train_lightgbm
from src.evaluate import evaluate_model, shap_analysis, get_risky_wallets
import pandas as pd
import os
import joblib

if __name__ == "__main__":
    # Загрузка и очистка
    df = load_and_clean_data("data/dataset.parquet")
    df = remove_high_corr_features(df)

    # EDA
    run_eda(df)

    # Разделение
    train, val, test = split_data(df, random_state=42)
    feature_cols = prepare_features(df)
    
    X_train_raw = train[feature_cols].copy()
    y_train = train['target'].copy()
    X_val_raw = val[feature_cols].copy()
    y_val = val['target'].copy()
    X_test_raw = test[feature_cols].copy()
    y_test = test['target'].copy()

    # Заполнение пропусков
    X_train, X_val, X_test = fill_missing_with_median(X_train_raw, X_val_raw, X_test_raw)

    # Отладочная информация
    print(" Оставленные признаки:", len(X_train.columns))
    print(X_train.columns.tolist())

    print(" Разделение завершено:")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Обучение
    model, best_params, feature_names_from_model = train_lightgbm(X_train, y_train)

    # Оценка с оптимальным порогом
    best_threshold = evaluate_model(
        model, 
        X_train, 
        X_val[feature_names_from_model], 
        X_test[feature_names_from_model], 
        y_train, y_val, y_test,
        name="LightGBM"
    )

    # SHAP и рискованные кошельки
    shap_analysis(model, X_test[feature_names_from_model], name="LightGBM")
    X_full = pd.concat([X_train, X_val, X_test])[feature_names_from_model]
    df_full = pd.concat([train, val, test]).reset_index(drop=True)
    get_risky_wallets(model, X_full, df_full, name="LightGBM")

    # ===========================================================
    #  СОХРАНЕНИЕ РЕФЕРЕНСНЫХ ДАННЫХ ДЛЯ МОНИТОРИНГА (PARQUET + CSV)
    # ===========================================================
    print(" Сохраняем референсные данные для мониторинга дрейфа...")
    os.makedirs("monitoring/reference", exist_ok=True)

    # Используем X_train как есть — он уже содержит только фичи, пошедшие в модель
    X_train_for_ref = X_train.copy()

    # Скоры на трейне
    train_scores = model.predict_proba(X_train_for_ref)[:, 1]

    # Parquet — для быстрой загрузки в скриптах
    X_train_for_ref.to_parquet("monitoring/reference/reference_features.parquet", index=False)
    pd.DataFrame({"score": train_scores}).to_parquet("monitoring/reference/reference_scores.parquet", index=False)

    # CSV — для удобного просмотра в VS Code / Excel
    X_train_for_ref.to_csv("monitoring/reference/reference_features.csv", index=False)
    pd.DataFrame({"score": train_scores}).to_csv("monitoring/reference/reference_scores.csv", index=False)

    print("Референсные данные сохранены в monitoring/reference/ (parquet + csv)")
    # Сохраняем порог
    joblib.dump(best_threshold, "models/lightgbm_best_threshold.pkl")
    print(f" Лучший порог сохранён: {best_threshold:.4f}")
    print(f"Использовано признаков: {len(feature_names_from_model)}")
    print("Обучение завершено!")

    print(" Пример входа для API:")
    print(X_test.iloc[0].to_dict())