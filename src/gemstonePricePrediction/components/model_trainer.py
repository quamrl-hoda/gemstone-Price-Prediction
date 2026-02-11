
import os
import sys
import logging
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.gemstonePricePrediction.entity.config_entity import ModelTrainerConfig
from src.gemstonePricePrediction.exception import CustomException
from src.gemstonePricePrediction.logger import logging
from src.gemstonePricePrediction.utils.common import save_object, evaluate_models, model_metrics, print_evaluated_results

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_training(self):

        try:
            logging.info("Loading transformed train and test arrays")

            train_array = np.load(self.config.train_array_path)
            test_array = np.load(self.config.test_array_path)

            xtrain, ytrain, xtest, ytest = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Arrays loaded successfully")

            models = {
                "Linear Regression": LinearRegression(**self.config.params["LinearRegression"]),
                "Lasso": Lasso(**self.config.params["Lasso"]),
                "Ridge": Ridge(**self.config.params["Ridge"]),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(**self.config.params["DecisionTreeRegressor"]),
                "Random Forest Regressor": RandomForestRegressor(**self.config.params["RandomForestRegressor"]),
                "XGBRegressor": XGBRegressor(**self.config.params["XGBRegressor"]),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor": GradientBoostingRegressor(**self.config.params["GradientBoostingRegressor"]),
                "AdaBoost Regressor": AdaBoostRegressor(**self.config.params["AdaBoostRegressor"])
            }

            logging.info("Evaluating multiple models")

            model_report: dict = evaluate_models(xtrain, ytrain, xtest, ytest, models)

            logging.info(f'Model Report : {model_report}')

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found', sys.exc_info())

            logging.info(
                f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}'
            )

            logging.info('Hyperparameter tuning started for CatBoost')

            cbr = CatBoostRegressor(verbose=False)

            param_dist = {
                "depth": self.config.params["CatBoostRegressor"]["depth"],
                "learning_rate": self.config.params["CatBoostRegressor"]["learning_rate"],
                "iterations": self.config.params["CatBoostRegressor"]["iterations"]
            }

            rscv = RandomizedSearchCV(
                cbr,
                param_dist,
                scoring='r2',
                cv=5,
                n_jobs=-1
            )

            rscv.fit(xtrain, ytrain)

            best_cbr = rscv.best_estimator_
            logging.info('Hyperparameter tuning complete for CatBoost')

            logging.info('Hyperparameter tuning started for KNN')
            
            # Use default KNN for grid search base or recreate
            knn = KNeighborsRegressor() 

            param_grid = {
                "n_neighbors": self.config.params["KNeighborsRegressor"]["n_neighbors"]
            }

            grid = GridSearchCV(
                knn,
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            grid.fit(xtrain, ytrain)

            best_knn = grid.best_estimator_
            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            er = VotingRegressor([
                ('cbr', best_cbr),
                ('xgb', XGBRegressor()),
                ('knn', best_knn)
            ], weights=[3, 2, 1])

            er.fit(xtrain, ytrain)
            
            print('Final Model Evaluation :\n')
            print_evaluated_results(xtrain, ytrain, xtest, ytest, er)

            logging.info('Voting Regressor Training Completed')

            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True
            )

            save_object(
                file_path=self.config.trained_model_file_path,
                obj=er
            )

            logging.info('Model pickle file saved')

            ytest_pred = er.predict(xtest)

            mae, rmse, r2 = model_metrics(ytest, ytest_pred)

            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')

            logging.info('Final Model Training Completed')

            return mae, rmse, r2

        except Exception as e:
            logging.error('Exception occurred at Model Training')
            raise CustomException(e, sys.exc_info())
