import joblib
import numpy as np 
def preprocess(Open, High, Low, Volume):
    model = joblib.load('model.pkl')
    features = np.array([[float(Open), float(High), float(Low), float(Volume)]])
    prediction = model.predict(features)
    return prediction
    