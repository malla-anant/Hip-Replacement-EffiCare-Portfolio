from flask import Flask, request, jsonify, render_template
from src.inference_pipeline import PredictionPipeline
import logging
import os

# ---------------------------------------------------
# App Initialization
# ---------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------
# Create Logs Folder (Safe for Deployment)
# ---------------------------------------------------

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------
# Load Prediction Pipeline Once
# ---------------------------------------------------

pipeline = PredictionPipeline()

# ---------------------------------------------------
# Home Page (Web UI)
# ---------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ---------------------------------------------------
# API Endpoint (For Postman / External Use)
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON input provided"
            }), 400

        logging.info(f"API Input: {data}")

        result = pipeline.predict(data)

        logging.info(f"API Prediction: {result}")

        return jsonify({
            "status": "success",
            "prediction": result
        })

    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------------------------
# Web Form Submission
# ---------------------------------------------------

@app.route("/web-predict", methods=["POST"])
def web_predict():
    try:
        data = request.form.to_dict()

        logging.info(f"Web Input (Raw): {data}")

        # Convert numeric fields safely
        numeric_int_fields = [
            "discharge_year",
            "ccs_diagnosis_code",
            "ccs_procedure_code",
            "apr_drg_code",
            "apr_mdc_code",
            "apr_severity_of_illness_code"
        ]

        numeric_float_fields = [
            "total_charges"
        ]

        for field in numeric_int_fields:
            if field in data and data[field] != "":
                data[field] = int(data[field])

        for field in numeric_float_fields:
            if field in data and data[field] != "":
                data[field] = float(data[field])

        logging.info(f"Web Input (Processed): {data}")

        result = pipeline.predict(data)

        logging.info(f"Web Prediction: {result}")

        return render_template("index.html", prediction=result)

    except Exception as e:
        logging.error(f"Web Error: {str(e)}")
        return render_template("index.html", error=str(e))


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False)
