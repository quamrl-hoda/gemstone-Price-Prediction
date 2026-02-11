import os
import sys
import logging
import zipfile
import pandas as pd

from urllib import request
from pathlib import Path
from sklearn.model_selection import train_test_split

from gemstonePricePrediction.entity.config_entity import DataIngestionConfig
from gemstonePricePrediction.exception import CustomException
from gemstonePricePrediction.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self) -> str:

        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )

                logging.info(
                    f"File: {filename} downloaded with following info: \n{headers}"
                )
            else:
                logging.info(
                    f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
                )

        except Exception as e:
            raise CustomException(e, sys)


    def extract_zip_file(self) -> None:

        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logging.info("Zip file extracted successfully")

        except Exception as e:
            raise CustomException(e, sys)


    # Data ingestion without raw data saving
    def initiate_data_ingestion(self):

        try:
            logging.info("Starting data ingestion process")

            # Read extracted dataset
            data_file_path = os.path.join(self.config.unzip_dir, "gemstone.csv")

            df = pd.read_csv(data_file_path)

            logging.info("Train Test Split Initiated")

            # Train Test Split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save train and test sets
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info("Ingestion of Data is completed")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
