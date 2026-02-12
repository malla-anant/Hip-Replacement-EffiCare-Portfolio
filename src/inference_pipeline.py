import joblib
import pandas as pd
import os
import logging


class PredictionPipeline:

    def __init__(self):
        base_path = "artifacts"

        self.los_model = joblib.load(os.path.join(base_path, "los_model.pkl"))
        self.cost_model = joblib.load(os.path.join(base_path, "cost_model.pkl"))
        self.readmission_model = joblib.load(os.path.join(base_path, "readmission_model.pkl"))

        # Expected feature names
        self.expected_features = list(self.los_model.feature_names_in_)

        logging.info(f"Expected Features: {self.expected_features}")

    # -------------------------------------------------------
    # Validate Input
    # -------------------------------------------------------
    def validate_input(self, input_dict):
        missing = [f for f in self.expected_features if f not in input_dict]

        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------
    def predict(self, input_dict):
        # ❌ DO NOT swallow exceptions silently
        self.validate_input(input_dict)

        df = pd.DataFrame([input_dict])

        # Predictions
        los = self.los_model.predict(df)[0]
        cost = self.cost_model.predict(df)[0]

        readmission_prob = self.readmission_model.predict_proba(df)[0][1]

        result = {
            "length_of_stay": round(float(los), 2),
            "estimated_cost": round(float(cost), 2),
            "readmission_probability": round(float(readmission_prob), 3),
            "risk_level": "High" if readmission_prob >= 0.5 else "Low"
        }

        logging.info(f"Prediction Result: {result}")

        return result
