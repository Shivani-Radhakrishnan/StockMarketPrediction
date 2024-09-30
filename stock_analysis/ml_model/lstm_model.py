import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def load_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler
