import pandas as pd
import os
from PhishingDetection.entity.config_entity import ModelTrainerConfig
from PhishingDetection import logger
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = load_npz(self.config.train_data_path)
        y_train = pd.read_csv(self.config.train_label_path).values.ravel()
        logger.info("Loaded X_train and y_train")

        lr = LogisticRegression(max_iter=self.config.max_iter)
        lr.fit(X_train, y_train)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
