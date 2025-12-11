from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, classification_report)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# внутри как тут нельзя делать from sklearn.metrics import (..., cross_val_score)  # ← НЕЛЬЗЯ!

 

import numpy as np
import pandas as pd
import os
import shap

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, name="LightGBM"):
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Предсказания вероятностей
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_val = model.predict_proba(X_val)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    # ROC AUC
    train_roc_auc = roc_auc_score(y_train, y_pred_train)
    val_roc_auc = roc_auc_score(y_val, y_pred_val)
    test_roc_auc = roc_auc_score(y_test, y_pred_test)
    print(f"Train ROC AUC: {train_roc_auc:.4f}")
    print(f"Val   ROC AUC: {val_roc_auc:.4f}")
    print(f"Test  ROC AUC: {test_roc_auc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"CV ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # === ОПТИМАЛЬНЫЙ ПОРОГ НА ОСНОВЕ F1 ===
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_test)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    print(f"\nBest threshold (by F1): {best_threshold:.4f}")

    # Классификация с лучшим порогом
    y_test_pred_class = (y_pred_test >= best_threshold).astype(int)
    report = classification_report(y_test, y_test_pred_class)
    print("\nClassification Report (Test) with best threshold:")
    print(report)

    # Сохраняем отчёт
    with open(f'results/{name}_classification_report.txt', 'w') as f:
        f.write(f"Best threshold: {best_threshold:.4f}\n\n")
        f.write(report)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC Curve')
    plt.legend()
    plt.savefig(f'plots/{name}_roc_curve.png')
    plt.close()

    # Precision-Recall Curve
    avg_precision = average_precision_score(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')
    plt.axvline(x=recall[best_threshold_idx], color='r', linestyle='--', label=f'Best thresh = {best_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name} Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'plots/{name}_pr_curve.png')
    plt.close()

    return best_threshold

def shap_analysis(model, X_test, name="LightGBM"):
    sample = X_test.sample(min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.savefig(f'plots/{name}_shap_importance.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(f'plots/{name}_shap_beeswarm.png', bbox_inches='tight')
    plt.close()

def get_risky_wallets(model, X_full, df_full, name="LightGBM"):
    probs = model.predict_proba(X_full)[:, 1]
    wallet_col = df_full.get('wallet_address', df_full.index)
    result = pd.DataFrame({
        'wallet': wallet_col,
        'probability': probs,
        'is_scammer': df_full['target']
    }).sort_values('probability', ascending=False).head(50)

    result.to_csv(f'results/{name}_top50_risky_wallets.csv', index=False)
