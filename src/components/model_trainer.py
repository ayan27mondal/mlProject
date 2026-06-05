import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()
    
  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Splitting the training and testing input data")
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1],
        train_array[:,-1],
        test_array[:,:-1],
        test_array[:,-1]
      )

      models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "XGBoost Regressor": XGBRegressor(),
        "CatBoost Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
        
      }
      
      # Hyperparameters for each model
      params = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },

    "Decision Tree": {
        "criterion": ["squared_error", "friedman_mse"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },

    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },

    "Linear Regression": {
        # No major hyperparameters to tune
    },

    "K-Neighbors Regressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },

    "XGBoost Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    },

    "CatBoost Regressor": {
        "iterations": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [4, 6, 8]
    },

    "AdaBoost Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    }
}
      

      model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
      
      # To get the best model score from the dictionary
      best_model_score = max(sorted(model_report.values()))
      
      # To get the best model name from the dictionary
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      
      best_model = models[best_model_name]
      
      if best_model_score < 0.6:
        raise CustomException("No best model found")
      logging.info(f"Best found model on both training and testing dataset is {best_model_name} with r2 score: {best_model_score}")
      
      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model,
      )
      
      prediction = best_model.predict(X_test)
      r2_square = r2_score(y_test, prediction)
      return r2_square
      
      
      
    except Exception as e:
      raise CustomException(e, sys)