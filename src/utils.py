# Common Functions
import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'wb') as file_obj:
      pickle.dump(obj, file_obj)

  except Exception as e:
    raise CustomException(e, sys)
  
def load_object(file_path):
  try:
    with open(file_path, 'rb') as file_obj:
      return pickle.load(file_obj)

  except Exception as e:
    raise CustomException(e, sys)
  
  
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
  try:
      report = {}

      for model_name, model in models.items():

          para = params[model_name]

          # If hyperparameters exist, use GridSearchCV
          if para:
              gs = GridSearchCV(
                  estimator=model,
                  param_grid=para,
                  cv=5,
                  n_jobs=-1
              )

              gs.fit(X_train, y_train)

              model = gs.best_estimator_

          # For models like Linear Regression
          else:
              model.fit(X_train, y_train)

          y_train_pred = model.predict(X_train)
          y_test_pred = model.predict(X_test)

          train_model_score = r2_score(y_train, y_train_pred)
          test_model_score = r2_score(y_test, y_test_pred)

          logging.info(
              f"{model_name} -> "
              f"Train R2: {train_model_score:.4f}, "
              f"Test R2: {test_model_score:.4f}"
          )

          report[model_name] = {
              "score": test_model_score,
              "model": model
          }

      return report

  except Exception as e:
      raise CustomException(e, sys)

  