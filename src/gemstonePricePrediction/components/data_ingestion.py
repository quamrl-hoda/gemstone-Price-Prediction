import os
import urllib.request as request
import zipfile
from gemstonePricePrediction.logger import logging
from gemstonePricePrediction.utils.common import get_size
from pathlib import Path
import urllib.request as request
from src.gemstonePricePrediction.entity.config_entity import DataIngestionConfig


class DataIngestion:
  def __init__(self, config: DataIngestionConfig):
      self.config = config

  def download_file(self,) -> str:
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

  def extract_zip_file(self,) -> None:
      """
      zip_file_path: str
      Extracts the zip file to the unzip directory
      Function return type: None
      """
      unzip_path = self.config.unzip_dir
      os.makedirs(unzip_path, exist_ok=True)
      with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
          zip_ref.extractall(unzip_path)
