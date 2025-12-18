FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка Python пакетов
RUN pip install --no-cache-dir numpy==1.24.3 opencv-python-headless==4.8.1.78 matplotlib pillow pyyaml fastapi uvicorn ultralytics==8.0.196

COPY app/ ./app/
COPY weights/ ./weights/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
