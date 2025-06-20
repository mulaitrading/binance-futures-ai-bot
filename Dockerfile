FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "binance_futures_bot_optimized_ai.py"]
