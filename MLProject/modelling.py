import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Autentikasi ke DagsHub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tianiayu/Membangun_model.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "tianiayu"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("HousePricePrediction")
mlflow.sklearn.autolog(disable=True)

# 2. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop(columns='price')
y_train = train_df['price']
X_test = test_df.drop(columns='price')
y_test = test_df['price']

# 3. Model Configs
model_configs = [
    {
        "name": "Linear Regression",
        "model": LinearRegression(),
        "params": {}
    },
    {
        "name": "Random Forest",
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        }
    },
    {
        "name": "Decision Tree",
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    }
]

os.makedirs('model', exist_ok=True)

# 4. Training Loop
for config in model_configs:
    name = config["name"]
    model = config["model"]
    param_grid = config["params"]

    print(f"\n Tuning {name}...")

    with mlflow.start_run(run_name=name) as run:
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
            print(f" Best Params for {name}: {grid.best_params_}")
        else:
            model.fit(X_train, y_train)
            best_model = model

        # Evaluation
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print(f" Evaluation for {name}:")
        print(f"   MAE  : {mae:.2f}")
        print(f"   RMSE : {rmse:.2f}")
        print(f"   R²   : {r2:.4f}")

        # Save model locally
        filename = f"model/{name.lower().replace(' ', '_')}_tuned.joblib"
        joblib.dump(best_model, filename)
        mlflow.log_artifact(filename)

        # Log model to MLflow tanpa model registry (aman untuk Dagshub)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_test[:1],
            registered_model_name=None 
        )


        print(f" Model {name} saved and logged at: model/")
