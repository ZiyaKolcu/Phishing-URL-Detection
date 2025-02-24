import os
import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from PhishingDetection import logger
from PhishingDetection.entity.config_entity import DataTransformationConfig
from scipy.sparse import save_npz


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def transform_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Data shape: {data.shape}")

        tokenizer = RegexpTokenizer(r"[A-Za-z]+")
        data["text_tokenized"] = data.url.map(lambda t: tokenizer.tokenize(t))
        logger.info("Text tokenized")

        stemmer = SnowballStemmer("english")
        data["text_stemmed"] = data["text_tokenized"].map(
            lambda l: [stemmer.stem(word) for word in l]
        )
        logger.info("Text stemmed")

        data["text_sent"] = data["text_stemmed"].map(lambda l: " ".join(l))
        logger.info("Text sent completed")

        cv = CountVectorizer()
        feature = cv.fit_transform(data.text_sent)

        X_train, X_test, y_train, y_test = train_test_split(
            feature, data.label, test_size=0.2, shuffle=True
        )
        logger.info("Train Test split completed")

        # Save sparse matrices in .npz format
        train_file_path = os.path.join(self.config.root_dir, "train.npz")
        test_file_path = os.path.join(self.config.root_dir, "test.npz")
        save_npz(train_file_path, X_train)
        save_npz(test_file_path, X_test)

        # Save labels separately in CSV format
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)

        logger.info("Splitted data into training and test sets.")
        logger.info(f"Train set shape: {X_train.shape}, Labels: {y_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}")
