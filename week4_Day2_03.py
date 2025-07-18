import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

# Set random seed for reproducibility
np.random.seed(42)

# Load and prepare data
data_path = r'C:\Users\游晨仪\Desktop\w2\US-pumpkins.csv'
pumpkin_data = pd.read_csv(data_path)
pumpkin_data = pumpkin_data.dropna(subset=['Low Price', 'High Price', 'Item Size', 'Origin', 'Variety', 'City Name'])
pumpkin_data['Price Range'] = pumpkin_data['High Price'] - pumpkin_data['Low Price']

features = ['Item Size', 'Origin', 'Variety', 'City Name', 'Package']
target = 'Low Price'
X = pumpkin_data[features]
y = pumpkin_data[target]

# Define models and encodings to test
models = [
    {
        "model_name": "LR",
        "model": LinearRegression(),
        "model_params": None
    },
    {
        "model_name": "RF",
        "model": RandomForestRegressor(n_estimators=100, random_state=42),
        "model_params": {"n_estimators": 100}
    }
]

encodings = [
    {
        "encoding_name": "onehot",
        "encoder": OneHotEncoder(handle_unknown='ignore')
    },
    {
        "encoding_name": "ordinal",
        "encoder": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    }
]

# Initialize KFold
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store all results
all_results = []

# Run experiments
for model_info in models:
    for encoding_info in encodings:
        # Create pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('encoder', encoding_info["encoder"], features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_info["model"])
        ])

        # Initialize fold results
        fold_results = []
        train_perfs = []
        test_perfs = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            pipeline.fit(X_train, y_train)

            # Predictions
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)

            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)

            # Store fold results
            fold_result = {
                f"{fold}_fold_train_data": [len(X_train), len(features)],
                f"{fold}_fold_test_data": [len(X_test), len(features)],
                f"{fold}_fold_train_performance": {
                    "rmse": f"{train_rmse:.2f}",
                    "mae": f"{train_mae:.2f}",
                    "r2": f"{train_r2:.2f}"
                },
                f"{fold}_fold_test_performance": {
                    "rmse": f"{test_rmse:.2f}",
                    "mae": f"{test_mae:.2f}",
                    "r2": f"{test_r2:.2f}"
                }
            }

            fold_results.append(fold_result)
            train_perfs.append((train_rmse, train_mae, train_r2))
            test_perfs.append((test_rmse, test_mae, test_r2))

        # Calculate average performance
        avg_train_rmse = np.mean([p[0] for p in train_perfs])
        avg_train_mae = np.mean([p[1] for p in train_perfs])
        avg_train_r2 = np.mean([p[2] for p in train_perfs])

        avg_test_rmse = np.mean([p[0] for p in test_perfs])
        avg_test_mae = np.mean([p[1] for p in test_perfs])
        avg_test_r2 = np.mean([p[2] for p in test_perfs])

        # Combine all results
        result = {
            "model_name": model_info["model_name"],
            "model_params": model_info["model_params"],
            "fea_encoding": encoding_info["encoding_name"],
            **fold_results[0],
            **fold_results[1],
            **fold_results[2],
            "average_train_performance": {
                "rmse": f"{avg_train_rmse:.2f}",
                "mae": f"{avg_train_mae:.2f}",
                "r2": f"{avg_train_r2:.2f}"
            },
            "average_test_performance": {
                "rmse": f"{avg_test_rmse:.2f}",
                "mae": f"{avg_test_mae:.2f}",
                "r2": f"{avg_test_r2:.2f}"
            }
        }

        all_results.append(result)

# Convert results to JSON and print
print(json.dumps(all_results, indent=2))