import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Setup MLflow
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/tianiayu/Membangun_model.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "tianiayu"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("HousePricePrediction")
mlflow.sklearn.autolog(disable=True)  # disable karena kita log manual

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
        "name": "Linear_Regression",
        "model": LinearRegression(),
        "params": {}
    },
    {
        "name": "Random_Forest",
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        }
    },
    {
        "name": "Decision_Tree",
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    }
]

os.makedirs('model', exist_ok=True)
best_run_id = None
best_mae = float('inf')

# 4. Training Loop
for config in model_configs:
    name = config["name"]
    print(f"\nTraining {name}...")

    with mlflow.start_run(run_name=name) as run:
        # Train model
        if config["params"]:
            grid = GridSearchCV(config["model"], config["params"], 
                                cv=3, scoring='neg_mean_absolute_error')
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
        else:
            model = config["model"]
            model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_run_id = run.info.run_id
        
        # Log metrics
        mlflow.log_metrics({
            "mae": mae,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        })

        # Save model (local & artifact)
        model_dir = "model"
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.save_model(
            sk_model=model,
            path=model_dir,
            signature=signature
        )

        mlflow.log_artifacts(model_dir, artifact_path="model")

        # Save locally for GitHub artifact
        joblib.dump(model, f"{model_dir}_{name}.joblib")

print(f"\nBest model run ID: {best_run_id}")
