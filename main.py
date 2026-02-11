from src.gemstonePricePrediction.logger import logging
from src.gemstonePricePrediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.gemstonePricePrediction.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.gemstonePricePrediction.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================x")
except Exception as e:
    logging.error(f"Error in stage {STAGE_NAME}: {e}")
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================x")
except Exception as e:
    logging.error(f"Error in stage {STAGE_NAME}: {e}")
    raise e


STAGE_NAME = "Data Transformation Stage"
if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================x")
    except Exception as e:
        logging.exception(e)
        raise e
