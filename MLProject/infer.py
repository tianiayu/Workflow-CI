import joblib
import pandas as pd

model = joblib.load("model.joblib")

for file in ["train.csv", "test.csv"]:
    data = pd.read_csv(file)
    pred = model.predict(data)
    pd.DataFrame({"prediction": pred}).to_csv(f"prediksi_{file}", index=False)
