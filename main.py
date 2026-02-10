from src.gemstonePricePrediction.logger import logging
from src.gemstonePricePrediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================x")
except Exception as e:
    logging.error(f"Error in stage {STAGE_NAME}: {e}")
    raise e
