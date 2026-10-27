import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [3, 5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [64, 128, 256],
                    "max_depth": [None, 10, 20],
                    "max_features": ["sqrt", "log2", None],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.6, 0.8, 0.9],
                    "n_estimators": [64, 128, 256],
                    "max_depth": [3, 5],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                    "max_depth": [3, 5, 7],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.1, 0.05, 0.01],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.5, 1.0],
                    "n_estimators": [50, 64, 128],
                },
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            # Best model score aur naam
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            # Sirf wahi parameters nikalo jo tune hue the
            best_params = {
                key: best_model.get_params()[key]
                for key in params.get(best_model_name, {})
            }

            # Sab models ka score (best se worst)
            print("\n===== Model Comparison (R2 score) =====")
            for name, score in sorted(model_report.items(), key=lambda x: x[1], reverse=True):
                print(f"{name:25s} : {score:.4f}")

            # Best model ki detail
            print("\n===== Best Model =====")
            print(f"Model      : {best_model_name}")
            print(f"R2 Score   : {best_model_score:.4f}")
            print(f"Best Params: {best_params}")

            logging.info(
                f"Best model: {best_model_name}, R2: {best_model_score}, Params: {best_params}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)