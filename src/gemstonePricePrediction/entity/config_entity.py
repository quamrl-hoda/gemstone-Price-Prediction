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




@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str
    mlflow_uri: str