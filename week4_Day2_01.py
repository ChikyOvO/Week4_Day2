import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib
import json
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

# 强制使用交互式后端（解决空白图像问题）
matplotlib.use('TkAgg')  # 也可以尝试 'Qt5Agg' 或其他可用后端

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 配置路径
class Config:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent
        self.DATA_PATH = Path(r"C:\Users\游晨仪\Desktop\w4\energy.csv")
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.create_dirs()

    def create_dirs(self):
        """创建必要的目录"""
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)


config = Config()


# 数据加载模块（增强数据检查）
class DataLoader:
    @staticmethod
    def load_data():
        """加载能源数据集"""
        try:
            logger.info(f"正在从 {config.DATA_PATH} 加载数据...")
            df = pd.read_csv(config.DATA_PATH)

            # 数据验证
            if df.empty:
                raise ValueError("加载的数据为空！")

            logger.info(f"数据加载成功，形状: {df.shape}")
            logger.info(f"前5行数据:\n{df.head()}")
            logger.info(f"数据类型:\n{df.dtypes}")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise


# 数据分析模块（修复图像显示）
class DataAnalyzer:
    @staticmethod
    def analyze(df):
        """执行基本的数据分析"""
        logger.info("开始数据分析...")

        analysis = {
            'head': df.head(),
            'describe': df.describe(),
            'null_values': df.isnull().sum(),
            'dtypes': df.dtypes,
            'correlation': df.select_dtypes(include=['number']).corr()
        }

        logger.info("数据分析完成")
        return analysis

    @staticmethod
    def visualize(df):
        """数据可视化（确保图像显示）"""
        logger.info("生成数据可视化...")

        # 数值特征的分布
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(num_cols) > 0:
            fig1, axes = plt.subplots(nrows=len(num_cols), figsize=(10, 2 * len(num_cols)))
            if len(num_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, num_cols):
                df[col].hist(ax=ax)
                ax.set_title(f"{col} 分布")

            plt.tight_layout()
            plt.savefig(config.OUTPUT_DIR / "distributions.png")
            plt.show()  # 确保显示
            plt.close(fig1)
        else:
            logger.warning("没有数值列可用于分布图")

        # 相关性热力图
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            fig2 = plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title("特征相关性热力图")
            plt.tight_layout()
            plt.savefig(config.OUTPUT_DIR / "correlation.png")
            plt.show()  # 确保显示
            plt.close(fig2)
        else:
            logger.warning("不足的数值列用于相关性热力图")


# 特征工程模块（保持不变）
class FeatureEngineer:
    @staticmethod
    def preprocess(df, target_col=None, encoding='ordinal'):
        """数据预处理"""
        logger.info("开始特征工程...")

        if target_col is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[-1]
                logger.warning(f"未指定目标列，自动选择: {target_col}")
            else:
                raise ValueError("数据中没有数值列可用作目标变量")

        if target_col not in df.columns:
            raise KeyError(f"目标列 '{target_col}' 不存在。可用列: {list(df.columns)}")

        datetime_cols = df.select_dtypes(include=['object']).columns
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df = df.drop(columns=[col])
            except:
                if encoding == 'ordinal':
                    df[col] = OrdinalEncoder().fit_transform(df[[col]])
                else:
                    df = pd.get_dummies(df, columns=[col])

        df = df.dropna()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        numeric_cols = X.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std()

        logger.info(f"特征工程完成 - 目标列: {target_col}")
        return X, y, target_col


# 建模模块（修复预测图显示）
class EnergyModel:
    def __init__(self, model_name='LR', model_params=None):
        self.model_name = model_name
        self.model_params = model_params

        if model_name == 'LR':
            self.model = LinearRegression(**model_params) if model_params else LinearRegression()
        elif model_name == 'LGBM':
            self.model = lgb.LGBMRegressor(**model_params) if model_params else lgb.LGBMRegressor()
        elif model_name == 'XGB':
            self.model = xgb.XGBRegressor(**model_params) if model_params else xgb.XGBRegressor()
        else:
            raise ValueError(f"未知模型类型: {model_name}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'rmse': f"{rmse:.2f}",
            'mae': f"{mae:.2f}",
            'r2': f"{r2:.2f}"
        }

        # 修复预测图显示
        fig = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.xlabel("真实值")
        plt.ylabel("预测值")
        plt.title(f"{self.model_name} 预测结果 vs 真实值")
        plt.tight_layout()
        plt.savefig(config.OUTPUT_DIR / f"{self.model_name}_predictions.png")
        plt.show()  # 确保显示
        plt.close(fig)

        return metrics


# 交叉验证模块
class CrossValidator:
    @staticmethod
    def cross_validate(model_name, X, y, n_splits=3, model_params=None, encoding='ordinal'):
        logger.info(f"开始 {model_name} 模型的 {n_splits} 折交叉验证...")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {
            "model_name": model_name,
            "model_params": model_params,
            "fea_encoding": encoding
        }

        fold_train_performances = []
        fold_test_performances = []

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            results[f"{i}_fold_train_data"] = [len(X_train), X_train.shape[1]]
            results[f"{i}_fold_test_data"] = [len(X_test), X_test.shape[1]]

            model = EnergyModel(model_name, model_params)
            model.train(X_train, y_train)

            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)

            results[f"{i}_fold_train_performance"] = train_metrics
            results[f"{i}_fold_test_performance"] = test_metrics

            fold_train_performances.append(train_metrics)
            fold_test_performances.append(test_metrics)

        # 计算平均性能
        avg_train = {
            'rmse': f"{np.mean([float(m['rmse']) for m in fold_train_performances]):.2f}",
            'mae': f"{np.mean([float(m['mae']) for m in fold_train_performances]):.2f}",
            'r2': f"{np.mean([float(m['r2']) for m in fold_train_performances]):.2f}"
        }

        avg_test = {
            'rmse': f"{np.mean([float(m['rmse']) for m in fold_test_performances]):.2f}",
            'mae': f"{np.mean([float(m['mae']) for m in fold_test_performances]):.2f}",
            'r2': f"{np.mean([float(m['r2']) for m in fold_test_performances]):.2f}"
        }

        results["average_train_performance"] = avg_train
        results["average_test_performance"] = avg_test

        # 保存结果到JSON文件
        result_file = config.RESULTS_DIR / f"{model_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"{model_name} 模型交叉验证完成，结果已保存到 {result_file}")
        return results


# 主流程
def main():
    try:
        logger.info("=== 能源数据分析系统启动 ===")

        data_loader = DataLoader()
        df = data_loader.load_data()

        analyzer = DataAnalyzer()
        analysis = analyzer.analyze(df)
        analyzer.visualize(df)

        # 特征工程
        engineer = FeatureEngineer()
        X, y, target_col = engineer.preprocess(df, encoding='ordinal')

        # 运行不同模型的交叉验证
        models = [
            ('LR', None),
            ('LGBM', None),
            ('XGB', None)
        ]

        all_results = []

        for model_name, params in models:
            results = CrossValidator.cross_validate(
                model_name=model_name,
                X=X,
                y=y,
                n_splits=3,
                model_params=params,
                encoding='ordinal'
            )
            all_results.append(results)

        logger.info("=== 所有模型评估完成 ===")
        return analysis, all_results, target_col

    except Exception as e:
        logger.error(f"主流程出错: {e}")
        raise


if __name__ == "__main__":
    try:
        analysis_results, model_results, target_col = main()
        print("\n分析结果摘要:")
        print(pd.DataFrame(analysis_results['describe']))
        print(f"\n目标列: {target_col}")

        print("\n模型评估结果:")
        for result in model_results:
            print(f"\n{result['model_name']} 模型:")
            print("平均训练性能:")
            print(pd.DataFrame.from_dict(result['average_train_performance'], orient='index'))
            print("\n平均测试性能:")
            print(pd.DataFrame.from_dict(result['average_test_performance'], orient='index'))
    except Exception as e:
        print(f"程序运行出错: {e}")