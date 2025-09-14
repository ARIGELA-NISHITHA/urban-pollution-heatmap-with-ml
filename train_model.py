# train_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Generate synthetic data (latitude, longitude -> pollution level)
np.random.seed(42)
latitudes = np.random.uniform(40.70, 40.80, 500)
longitudes = np.random.uniform(-74.02, -73.90, 500)
pollution = (np.sin(latitudes * 10) + np.cos(longitudes * 10)) * 20 + 50 + np.random.normal(0, 5, 500)

data = pd.DataFrame({
    'latitude': latitudes,
    'longitude': longitudes,
    'pollution': pollution
})

X = data[['latitude', 'longitude']]
y = data['pollution']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'pollution_model.pkl')
print("Model trained and saved!")
