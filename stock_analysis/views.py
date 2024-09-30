from django.shortcuts import render
from keras.models import load_model
from ml_model.lstm_model import load_stock_data, preprocess_data

def predict_stock(request):
    if request.method == 'POST':
        stock_symbol = request.POST.get('symbol')
        model = load_model('ml_model/lstm_stock_model.h5')

        stock_data = load_stock_data(stock_symbol, '2022-01-01', '2023-01-01')
        scaled_data, scaler = preprocess_data(stock_data)

        predicted_stock_price = model.predict(scaled_data.reshape(1, -1, 1))
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        
        return render(request, 'result.html', {'predicted_price': predicted_stock_price[0][0]})
    return render(request, 'predict.html')
