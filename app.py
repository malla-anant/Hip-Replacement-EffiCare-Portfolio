from flask import Flask, request, jsonify, render_template
from src.inference_pipeline import PredictionPipeline
import logging
import os

app = Flask(__name__)

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------
# Load Model Pipeline ONCE
# ---------------------------------------------------
pipeline = PredictionPipeline()

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ---------------------------------------------------
# API Endpoint (JSON)
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        result = pipeline.predict(data)

        return jsonify({
            "status": "success",
            "prediction": result
        })

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------
# Web Prediction
# ---------------------------------------------------
@app.route("/web-predict", methods=["POST"])
def web_predict():
    try:
        data = request.form.to_dict()

        # Convert numeric fields
        int_fields = [
            "discharge_year",
            "ccs_diagnosis_code",
            "ccs_procedure_code",
            "apr_drg_code",
            "apr_mdc_code",
            "apr_severity_of_illness_code"
        ]

        float_fields = ["total_charges"]

        for f in int_fields:
            if f in data and data[f]:
                data[f] = int(data[f])

        for f in float_fields:
            if f in data and data[f]:
                data[f] = float(data[f])

        result = pipeline.predict(data)

        return render_template("index.html", prediction=result)

    except Exception as e:
        logging.error(str(e))
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=False)
