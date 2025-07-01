import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from dagshub import dagshub_logger
import dagshub
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Autentikasi ke DagsHub
dagshub.init(
    repo_owner='tianiayu',
    repo_name='Membangun_model',
    token=os.getenv('DAGSHUB_TOKEN'),
    mlflow=True
)

# 1. Set tracking URI dan nama eksperimen
mlflow.set_tracking_uri("https://dagshub.com/tianiayu/Membangun_model.mlflow")
mlflow.set_experiment("HousePricePrediction")
mlflow.sklearn.autolog()

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

# 4. Training & Logging
os.makedirs('model', exist_ok=True)

for config in model_configs:
    name = config["name"]
    model = config["model"]
    param_grid = config["params"]

    print(f"\n Tuning {name}...")

    with mlflow.start_run(run_name=name):
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f" Best Params for {name}: {grid.best_params_}")
            mlflow.log_params(grid.best_params_)
        else:
            model.fit(X_train, y_train)
            best_model = model

        # Evaluasi
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f" Evaluation for {name}:")
        print(f"   MAE  : {mae:.2f}")
        print(f"   RMSE : {rmse:.2f}")
        print(f"   R²   : {r2:.4f}")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Simpan model
        filename = f"model/{name.lower().replace(' ', '_')}_tuned.joblib"
        joblib.dump(best_model, filename)
        mlflow.log_artifact(filename)
        print(f" Saved best {name} model to {filename}")
