import os
import pickle
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)
