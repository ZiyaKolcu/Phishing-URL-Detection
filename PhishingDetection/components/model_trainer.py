import pandas as pd
import os
from PhishingDetection.entity.config_entity import ModelTrainerConfig
from PhishingDetection import logger
from sklearn.ensemble import RandomForestClassifier
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]].values.ravel()

        classifier = RandomForestClassifier(
            n_estimators=self.config.n_estimators, max_features=self.config.max_features
        )
        classifier.fit(train_x, train_y)

        joblib.dump(
            classifier, os.path.join(self.config.root_dir, self.config.model_name)
        )
