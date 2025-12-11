# monitoring/retrain_if_needed.py
import subprocess
import sys

def retrain_if_needed():
    print("🔍 Проверка необходимости переобучения...")
    
    from monitoring.check_data_drift import check_data_drift
    from monitoring.check_score_drift import check_score_drift

    feature_drift = check_data_drift()
    score_drift = check_score_drift()
    
    if feature_drift or score_drift:
        print("🔄 Запуск переобучения модели...")
        result = subprocess.run([sys.executable, "train_pipeline.py"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Модель успешно переобучена!")
        else:
            print("❌ Ошибка при переобучении:")
            print(result.stderr)
    else:
        print("ℹ️ Переобучение не требуется")

if __name__ == "__main__":
    retrain_if_needed()