import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

class TrainPipeline:
  def __init__(self):
    pass
  
  def run_pipeline(self):
    try:
      data_ingestion = DataIngestion()
      train_data, test_data = data_ingestion.initiate_data_ingestion()
      
      data_transformation = DataTransformation()
      train_array, test_array,_= data_transformation.initiate_data_transformation(train_data, test_data)
      
      model_trainer = ModelTrainer()
      r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
      
      logging.info(f"Model Training Completed. R2 Score: {r2_score}")

      print(f"\nModel Training Completed")
      print(f"R2 Score = {r2_score:.4f}")
      
    except Exception as e:
      raise CustomException(e, sys)
    
if __name__ == "__main__":
  obj = TrainPipeline()
  obj.run_pipeline()