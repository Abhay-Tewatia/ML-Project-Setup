import os
import pickle
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from  sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for name, model in models.items():
            param_grid = params.get(name, {})

            if param_grid:
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=20,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                )
                search.fit(X_train, y_train)
                model.set_params(**search.best_params_)  # apply best params

            model.fit(X_train, y_train)  # final fit with the best params

            y_test_pred = model.predict(X_test)
            report[name] = r2_score(y_test, y_test_pred)

        return report

    except Exception as e:
        logging.error(f"Error occurred while evaluating models: {str(e)}")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)