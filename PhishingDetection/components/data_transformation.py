import os
import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PhishingDetection import logger
from PhishingDetection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(f"Data shape: {data.shape}")

        data["url"] = data["url"].replace("www.", "", regex=True)

        lb_make = LabelEncoder()
        data["class_url"] = lb_make.fit_transform(data["label"])

        data["url_len"] = data["url"].apply(lambda x: len(str(x)))

        data["abnormal_url"] = data["url"].apply(
            lambda URL: 1 if re.search(str(urlparse(URL).hostname), URL) else 0
        )

        pattern_ip = (
            r"(([01]?\d\d?|2[0-4]\d|25[0-5])\."
            r"([01]?\d\d?|2[0-4]\d|25[0-5])\."
            r"([01]?\d\d?|2[0-4]\d|25[0-5])\."
            r"([01]?\d\d?|2[0-4]\d|25[0-5])\/)|"
            r"((0x[0-9a-fA-F]{1,2})\."
            r"(0x[0-9a-fA-F]{1,2})\."
            r"(0x[0-9a-fA-F]{1,2})\."
            r"(0x[0-9a-fA-F]{1,2})\/)"
            r"(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}"
        )
        data["use_of_ip_address"] = data["url"].apply(
            lambda URL: 1 if re.search(pattern_ip, URL) else 0
        )

        special_chars = [
            "@",
            "?",
            "-",
            "=",
            ".",
            "#",
            "%",
            "+",
            "$",
            "!",
            "*",
            ",",
            "//",
        ]
        data["sum_count_special_chars"] = data["url"].apply(
            lambda x: sum(char in x for char in special_chars)
        )

        data["https"] = data["url"].apply(
            lambda x: 1 if urlparse(x).scheme == "https" else 0
        )

        data["digits"] = data["url"].apply(
            lambda x: sum(1 for char in x if char.isnumeric())
        )

        data["letters"] = data["url"].apply(
            lambda x: sum(1 for char in x if char.isalpha())
        )

        pattern_shortening = (
            r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
            r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
            r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
            r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|"
            r"db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|"
            r"q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
            r"x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
            r"tr\.im|link\.zip\.net"
        )
        data["Shortining_Service"] = data["url"].apply(
            lambda x: 1 if re.search(pattern_shortening, x) else 0
        )

        columns_to_drop = ["url", "label", "class_url"]
        X = data.drop(columns=columns_to_drop, errors="ignore")
        y = data["class_url"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=5
        )

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_file_path = os.path.join(self.config.root_dir, "train.csv")
        test_file_path = os.path.join(self.config.root_dir, "test.csv")
        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)

        logger.info("Splited data into training and test sets.")
        logger.info(f"Train set shape: {train_data.shape}")
        logger.info(f"Test set shape: {test_data.shape}")
