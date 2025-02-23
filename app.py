from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import re
import pandas as pd
from urllib.parse import urlparse

app = Flask(__name__)


MODEL_PATH = "artifacts/model_trainer/model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")


def extract_features(url):
    features = {}
    features["url_len"] = len(url)
    features["abnormal_url"] = 1 if re.search(str(urlparse(url).hostname), url) else 0

    pattern_ip = (
        r"(([01]?\d\d?|2[0-4]\d|25[0-5])\."
        r"([01]?\d\d?|2[0-4]\d|25[0-5])\."
        r"([01]?\d\d?|2[0-4]\d|25[0-5])\."
        r"([01]?\d\d?|2[0-4]\d|25[0-5])\/)|"
        r"((0x[0-9a-fA-F]{1,2})\."
        r"(0x[0-9a-fA-F]{1,2})\."
        r"(0x[0-9a-fA-F]{1,2})\."
        r"(0x[0-9a-fA-F]{1,2})\/)"
    )
    features["use_of_ip_address"] = 1 if re.search(pattern_ip, url) else 0

    special_chars = ["@", "?", "-", "=", ".", "#", "%", "+", "$", "!", "*", ",", "//"]
    features["sum_count_special_chars"] = sum(char in url for char in special_chars)

    features["https"] = 1 if urlparse(url).scheme == "https" else 0
    features["digits"] = sum(1 for char in url if char.isnumeric())
    features["letters"] = sum(1 for char in url if char.isalpha())

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
    features["Shortining_Service"] = 1 if re.search(pattern_shortening, url) else 0

    return np.array(list(features.values())).reshape(1, -1)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form["url"]
        features = extract_features(url)
        prediction = model.predict(features)
        return render_template(
            "index.html", prediction_text=f"Prediction: {prediction[0]}"
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
