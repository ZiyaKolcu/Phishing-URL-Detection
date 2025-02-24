from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)


MODEL_PATH = "ml_model/model.joblib"
CV_PATH = "ml_model/cv.joblib"
if os.path.exists(MODEL_PATH) and os.path.exists(CV_PATH):
    model = joblib.load(MODEL_PATH)
    cv = joblib.load(CV_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' or '{CV_PATH}' not found.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        url = request.form["url"]
        transformed_url = cv.transform([url])
        prediction = model.predict(transformed_url)

        css_class = "text-success" if prediction[0] == "good" else "text-danger"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {prediction[0]}",
            css_class=css_class,
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
