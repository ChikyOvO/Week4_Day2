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
import matplotlib.pyplot as plt


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载和预处理
data_path = r'C:\Users\游晨仪\Desktop\w2\US-pumpkins.csv'
pumpkin_data = pd.read_csv(data_path)
pumpkin_data = pumpkin_data.dropna(subset=['Low Price', 'High Price', 'Item Size', 'Origin', 'Variety', 'City Name'])
pumpkin_data['Price Range'] = pumpkin_data['High Price'] - pumpkin_data['Low Price']

features = ['Item Size', 'Origin', 'Variety', 'City Name', 'Package']
target = 'Low Price'

X = pumpkin_data[features]
y = pumpkin_data[target]


# 定义评估函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        "rmse": f"{rmse:.2f}",
        "mae": f"{mae:.2f}",
        "r2": f"{r2:.2f}"
    }


# 定义交叉验证和结果记录函数
def run_experiment(model_name, model, fea_encoding, X, y, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        "model_name": model_name,
        "model_params": str(model.get_params()) if model_name == "RF" else None,
        "fea_encoding": fea_encoding,
    }

    train_perfs = []
    test_perfs = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 记录数据形状
        results[f"{i}_fold_train_data"] = list(X_train.shape)
        results[f"{i}_fold_test_data"] = list(X_test.shape)

        # 训练模型
        model.fit(X_train, y_train)

        # 评估训练集和测试集
        train_perf = evaluate_model(model, X_train, y_train)
        test_perf = evaluate_model(model, X_test, y_test)

        results[f"{i}_fold_train_performance"] = train_perf
        results[f"{i}_fold_test_performance"] = test_perf

        train_perfs.append(train_perf)
        test_perfs.append(test_perf)

    # 计算平均性能
    avg_train = {
        "rmse": f"{np.mean([float(p['rmse']) for p in train_perfs]):.2f}",
        "mae": f"{np.mean([float(p['mae']) for p in train_perfs]):.2f}",
        "r2": f"{np.mean([float(p['r2']) for p in train_perfs]):.2f}"
    }

    avg_test = {
        "rmse": f"{np.mean([float(p['rmse']) for p in test_perfs]):.2f}",
        "mae": f"{np.mean([float(p['mae']) for p in test_perfs]):.2f}",
        "r2": f"{np.mean([float(p['r2']) for p in test_perfs]):.2f}"
    }

    results["average_train_performance"] = avg_train
    results["average_test_performance"] = avg_test

    return results


# 定义预处理管道
numeric_features = []
categorical_features = features  # 所有特征都是分类的

# 两种编码方式
preprocessor_ohe = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

preprocessor_ord = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ])

# 定义模型
linear_model_ohe = Pipeline(steps=[
    ('preprocessor', preprocessor_ohe),
    ('regressor', LinearRegression())
])

linear_model_ord = Pipeline(steps=[
    ('preprocessor', preprocessor_ord),
    ('regressor', LinearRegression())
])

rf_model_ohe = Pipeline(steps=[
    ('preprocessor', preprocessor_ohe),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_model_ord = Pipeline(steps=[
    ('preprocessor', preprocessor_ord),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 运行所有实验
experiments = [
    ("LR", linear_model_ohe, "onehot"),
    ("LR", linear_model_ord, "ordinal"),
    ("RF", rf_model_ohe, "onehot"),
    ("RF", rf_model_ord, "ordinal")
]

all_results = []
for model_name, model, encoding in experiments:
    result = run_experiment(model_name, model, encoding, X, y)
    all_results.append(result)

# 输出结果
print(json.dumps(all_results, indent=2))