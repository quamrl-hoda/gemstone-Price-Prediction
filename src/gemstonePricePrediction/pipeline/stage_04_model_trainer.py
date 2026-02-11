from src.gemstonePricePrediction.config.configuration import ConfigurationManager
from src.gemstonePricePrediction.components.model_trainer import ModelTrainer
from src.gemstonePricePrediction.logger import logging
import sys
from src.gemstonePricePrediction.exception import CustomException


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logging.info("=" * 30)
            logging.info("Stage 4: Model Training Pipeline Started")
            logging.info("=" * 30)

            # Initialize Configuration Manager
            config_manager = ConfigurationManager()

            # Get Model Trainer Configuration
            model_trainer_config = config_manager.get_model_trainer_config()

            # Initialize Model Trainer
            model_trainer = ModelTrainer(config=model_trainer_config)

            # Run Model Training
            model_trainer.initiate_model_training()

            logging.info("=" * 30)
            logging.info("Stage 4: Model Training Pipeline Completed Successfully")
            logging.info("=" * 30)

        except Exception as e:
            logging.error(f"Error in Model Training Pipeline: {e}")
            raise CustomException(e, sys)   