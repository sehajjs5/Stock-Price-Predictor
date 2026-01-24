from flask import Flask, render_template, request
import utils
import json

app = Flask(__name__)

STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "JPM": "JP Morgan",
    "BAC": "Bank of America",
    "RELIANCE.NS": "Reliance",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank"
}

@app.route('/')
def home():
    return render_template("index.html", stocks=STOCKS)

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    mode = request.form['mode']
    
    if mode=='custom':
        open = float(request.form['open'])
        volume = float(request.form['volume'])
        prediction, history = utils.predict_custom(ticker, open, volume)
    else:
        prediction, history = utils.predict_stock(ticker)

    return render_template(
        "prediction.html",
        prediction=prediction,
        ticker=ticker,
        graph_data=json.dumps(history),
        mode = mode
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
