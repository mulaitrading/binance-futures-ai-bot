# Binance Futures Trading Bot with AI LSTM

This project is a Binance Futures trading bot using LSTM AI prediction, integrated with technical indicators (RSI, EMA, MACD), auto TP/SL, trailing stop, and Telegram notifications.

## Environment Variables
Create a `.env` file with the following:

```
API_KEY=YOUR_BINANCE_API_KEY
API_SECRET=YOUR_BINANCE_API_SECRET
TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID
```

## Deployment on Railway
1. Connect this repo to Railway → https://railway.app
2. Set the environment variables in Railway dashboard.
3. Deploy → Railway will automatically detect Dockerfile and build the project.
