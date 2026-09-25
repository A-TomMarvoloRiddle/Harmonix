FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ml/ ./backend/ml/

# In a real setup, this would run a celery worker or a kafka consumer loop
CMD ["python", "-m", "backend.ml.train_two_tower"]
