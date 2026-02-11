from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    source_URL: str
    local_data_file: str
    unzip_dir: str
    train_data_path: str
    test_data_path: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass
class DataTransformationConfig:
    root_dir: str
    train_data_path: str
    test_data_path: str
    preprocessor_obj_file_path: str



@dataclass
class ModelTrainerConfig:
    root_dir: str
    trained_model_file_path: str
    train_array_path: str
    test_array_path: str
    target_column: str
    params: dict




# @dataclass(frozen=True)
# class ModelEvaluationConfig:
#     root_dir: Path
#     test_data_path: Path
#     model_path: Path
#     metric_file_name: Path
#     target_column: str
#     mlflow_uri: str