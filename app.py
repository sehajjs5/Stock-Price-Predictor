from flask import Flask, render_template, request
from logging import debug
import utils
import json

app = Flask(__name__)
@app.route('/')
def home(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    Open = request.form.get("Open")
    High = request.form.get("High")
    Low = request.form.get("Low")
    Volume = request.form.get("Volume")
    prediction = utils.preprocess(Open, High, Low, Volume)
    return render_template('prediction.html', prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)
