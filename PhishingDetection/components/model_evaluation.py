import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from PhishingDetection.entity.config_entity import ModelEvaluationConfig
from PhishingDetection.constants import *
from PhishingDetection.utils.common import save_json
from PhishingDetection import logger
from scipy.sparse import load_npz
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, pos_label="good")
        recall = recall_score(actual, pred, pos_label="good")
        f1 = f1_score(actual, pred, pos_label="good")
        return accuracy, precision, recall, f1

    def log_into_mlflow(self):
        model = joblib.load(self.config.model_path)
        logger.info("Loaded ML model")

        X_test = load_npz(self.config.test_data_path)
        y_test = pd.read_csv(self.config.test_label_path).values.ravel()
        logger.info("Loaded X_test and y_test")

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_labels = model.predict(X_test)
            accuracy, precision, recall, f1 = self.eval_metrics(
                y_test, predicted_labels
            )

            # Saving metrics as local
            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="LogisticRegression"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
