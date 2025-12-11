# Используем официальный Python-образ
FROM python:3.11-slim

# Устанавливаем зависимости
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Порт для API
EXPOSE 5000

# Запуск API
CMD ["python", "app/api.py"]