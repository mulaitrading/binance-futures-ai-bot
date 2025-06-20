
import numpy as np
import pandas as pd
import time
from binance.client import Client
from binance.enums import *
import httpx

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'

SYMBOL = 'BTCUSDT'
QTY = 0.001
EMA_PERIOD = 20
TP_PERCENT = 1.5
SL_PERCENT = 0.8
TRAILING_PERCENT = 0.7
SEQ_LEN = 60
MODEL_FILE = "lstm_optimized_model.h5"

client = Client(API_KEY, API_SECRET)

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    httpx.post(url, data=payload)

def get_price():
    return float(client.futures_symbol_ticker(symbol=SYMBOL)['price'])

def place_order(side, qty, tp_percent, sl_percent):
    order = client.futures_create_order(symbol=SYMBOL, side=side, type=ORDER_TYPE_MARKET, quantity=qty)
    entry_price = float(order['fills'][0]['price'])
    tp_price = entry_price * (1 + tp_percent / 100) if side == SIDE_BUY else entry_price * (1 - tp_percent / 100)
    sl_price = entry_price * (1 - sl_percent / 100) if side == SIDE_BUY else entry_price * (1 + sl_percent / 100)

    client.futures_create_order(symbol=SYMBOL, side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                                type=ORDER_TYPE_LIMIT, quantity=qty, price=round(tp_price, 2), timeInForce=TIME_IN_FORCE_GTC)

    client.futures_create_order(symbol=SYMBOL, side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                                type=ORDER_TYPE_STOP_MARKET, stopPrice=round(sl_price, 2), closePosition=True)

    send_telegram(f"Order {side} {SYMBOL} @ {entry_price}\nTP: {tp_price}\nSL: {sl_price}")

def trailing_stop(entry_price, side, trailing_percent):
    while True:
        price = get_price()
        profit = (price - entry_price) / entry_price * 100 if side == SIDE_BUY else (entry_price - price) / entry_price * 100
        print(f"Trailing Profit: {profit:.2f}%")
        if profit >= trailing_percent:
            client.futures_create_order(symbol=SYMBOL, side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                                        type=ORDER_TYPE_MARKET, quantity=QTY)
            send_telegram(f"Trailing Stop executed @ {price}")
            break
        time.sleep(5)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_historical_prices(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_15MINUTE, lookback="1000 hours ago UTC"):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['close', 'volume']]

def prepare_dataset_with_indicators():
    df = get_historical_prices()
    df['rsi'] = calculate_rsi(df['close'])
    df['ema'] = calculate_ema(df['close'])
    macd, signal = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i-SEQ_LEN:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def build_optimized_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_ai_model():
    X, y, scaler = prepare_dataset_with_indicators()
    model = build_optimized_lstm((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)
    model.save(MODEL_FILE)
    return model, scaler

def predict_next_price(model, scaler):
    df = get_historical_prices()
    df['rsi'] = calculate_rsi(df['close'])
    df['ema'] = calculate_ema(df['close'])
    macd, signal = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df.dropna(inplace=True)

    scaled = scaler.transform(df)
    last_seq = scaled[-SEQ_LEN:, :].reshape(1, -1, scaled.shape[1])
    predicted_scaled = model.predict(last_seq)[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled] + [0]*(scaled.shape[1]-1)])[0][0]
    return predicted_price

price_history = []
try:
    try:
        model = load_model(MODEL_FILE)
        print("Loaded pre-trained LSTM model.")
        _, _, scaler = prepare_dataset_with_indicators()
    except:
        print("Training new model...")
        model, scaler = train_ai_model()

    while True:
        price = get_price()
        price_history.append(price)
        if len(price_history) > EMA_PERIOD + 5:
            price_history.pop(0)

        ema = pd.Series(price_history).ewm(span=EMA_PERIOD, adjust=False).mean().values[-1]
        ai_prediction = predict_next_price(model, scaler)

        print(f"Price: {price} | EMA({EMA_PERIOD}): {ema:.2f} | AI Predict: {ai_prediction:.2f}")

        if price > ema and ai_prediction > price:
            place_order(SIDE_BUY, QTY, TP_PERCENT, SL_PERCENT)
            trailing_stop(price, SIDE_BUY, TRAILING_PERCENT)

        elif price < ema and ai_prediction < price:
            place_order(SIDE_SELL, QTY, TP_PERCENT, SL_PERCENT)
            trailing_stop(price, SIDE_SELL, TRAILING_PERCENT)

        time.sleep(15)

except KeyboardInterrupt:
    print("Bot dihentikan manual")
