import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load("model.joblib")

# Dummy input dengan jumlah fitur sesuai model
dummy_input = pd.DataFrame([np.zeros(model.n_features_in_)])
prediction = model.predict(dummy_input)

print("Prediction:", prediction)
