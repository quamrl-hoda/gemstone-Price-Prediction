from src.gemstonePricePrediction.config.configuration import ConfigurationManager
from src.gemstonePricePrediction.components.data_ingestion import DataIngestion
from src.gemstonePricePrediction.logger import logging

STAGE_NAME = "Data Ingestion Stage"
class DataIngestionTrainingPipeline:
  def __init__(self):
    pass

  def main(self):
    try:
      config = ConfigurationManager()
      data_ingestion_config = config.get_data_ingestion_config()

      data_ingestion = DataIngestion(config=data_ingestion_config)

      data_ingestion.download_file()
      data_ingestion.extract_zip_file()

      train_path, test_path = data_ingestion.initiate_data_ingestion()

      print("Train file created at:", train_path)
      print("Test file created at:", test_path)

    except Exception as e:
      raise e
