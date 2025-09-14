# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('pollution_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = data.get('latitude')
    lon = data.get('longitude')
    pred = model.predict(np.array([[lat, lon]]))[0]
    return jsonify({'pollution': pred})

if __name__ == "__main__":
    app.run(debug=True)
