import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Aktifkan autolog
mlflow.sklearn.autolog()

# Load dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Split fitur dan label
X_train = train_data.drop("price", axis=1)
y_train = train_data["price"]
X_test = test_data.drop("price", axis=1)
y_test = test_data["price"]

# Jalankan MLflow experiment
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)

    # Log eksplisit (tidak wajib karena ada autolog, tapi lebih aman)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Simpan model
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")
