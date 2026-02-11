
from src.gemstonePricePrediction.config.configuration import ConfigurationManager
from src.gemstonePricePrediction.components.data_transformation import DataTransformation
from src.gemstonePricePrediction.logger import logging
import sys

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.initiate_data_transformation()
        except Exception as e:
            raise e

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx================x")
    except Exception as e:
        logging.exception(e)
        raise e
