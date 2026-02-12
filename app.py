from flask import Flask, request, jsonify, render_template
from src.inference_pipeline import PredictionPipeline
import logging
import os

# ---------------------------------------------------
# App Initialization
# ---------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------
# Logging (Safe for Render)
# ---------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------
# Load ML Pipeline ONCE
# ---------------------------------------------------
pipeline = PredictionPipeline()

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ---------------------------------------------------
# Web Form Prediction
# ---------------------------------------------------
@app.route("/web-predict", methods=["POST"])
def web_predict():
    try:
        data = request.form.to_dict()

        logging.info(f"Web Input Raw: {data}")

        # Convert numeric values
        int_fields = [
            "discharge_year",
            "ccs_diagnosis_code",
            "ccs_procedure_code",
            "apr_drg_code",
            "apr_mdc_code",
            "apr_severity_of_illness_code"
        ]

        float_fields = ["total_charges"]

        for field in int_fields:
            data[field] = int(data[field])

        for field in float_fields:
            data[field] = float(data[field])

        prediction = pipeline.predict(data)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return render_template(
            "index.html",
            error=str(e)
        )

# ---------------------------------------------------
# API Endpoint (Optional)
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        result = pipeline.predict(data)
        return jsonify({"status": "success", "prediction": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
