from src.gemstonePricePrediction.logger import logging
from src.gemstonePricePrediction.config.configuration import ConfigurationManager
from src.gemstonePricePrediction.components.data_validation import DataValiadtion




STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValiadtion(config=data_validation_config)
            data_validation.validate_all_columns()

            logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================x")
        except Exception as e:
            logging.error(f"Error in stage {STAGE_NAME}: {e}")
            raise e