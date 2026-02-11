
import sys
import os
import pandas as pd
import numpy as np
from src.gemstonePricePrediction.exception import CustomException
from src.gemstonePricePrediction.utils.common import load_object

class PredictionPipeline:
    def __init__(self):
        # Hardcoded paths based on project structure
        self.model_path = os.path.join("models", "model.pkl")
        self.preprocessor_path = os.path.join("models", "preprocessor.pkl")

    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            
            logging.info("Modeling and Preprocessor loaded")
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

# For logging purposes within this file
from src.gemstonePricePrediction.logger import logging