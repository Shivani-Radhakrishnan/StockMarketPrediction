from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from ml_model.lstm_model import load_stock_data, preprocess_data
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model once when the app starts
model = load_model('ml_model/lstm_stock_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_stock():
    stock_symbol = request.form['symbol']
    prediction_date = request.form['date']

    # Load and preprocess stock data
    stock_data = load_stock_data(stock_symbol, '2022-01-01', '2023-01-01')
    scaled_data, scaler = preprocess_data(stock_data)

    # Predicting stock price
    predicted_stock_price = model.predict(scaled_data.reshape(1, -1, 1))
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    return render_template('result.html', predicted_price=predicted_stock_price[0][0], stock_symbol=stock_symbol, prediction_date=prediction_date)

if __name__ == '__main__':
    app.run(debug=True)
