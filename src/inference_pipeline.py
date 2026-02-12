import joblib
import pandas as pd
import os


class PredictionPipeline:

    def __init__(self):

        base_path = "artifacts"

        self.los_model = joblib.load(os.path.join(base_path, "los_model.pkl"))
        self.cost_model = joblib.load(os.path.join(base_path, "cost_model.pkl"))
        self.readmission_model = joblib.load(os.path.join(base_path, "readmission_model.pkl"))

        # Store expected feature names from LOS model preprocessor
        self.expected_features = self.los_model.feature_names_in_

    # -------------------------------------------------------
    # Validate Input
    # -------------------------------------------------------

    def validate_input(self, input_dict):

        missing_features = [
            feature for feature in self.expected_features
            if feature not in input_dict
        ]

        if missing_features:
            raise ValueError(f"Missing required fields: {missing_features}")

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------

    def predict(self, input_dict):

        try:
            # Validate input fields
            self.validate_input(input_dict)

            # Convert to DataFrame
            df = pd.DataFrame([input_dict])

            # Predictions
            los = self.los_model.predict(df)[0]
            cost = self.cost_model.predict(df)[0]

            readmission_prob = self.readmission_model.predict_proba(df)[0][1]
            readmission_class = 1 if readmission_prob >= 0.5 else 0

            result = {
                "Predicted_Length_of_Stay": round(float(los), 2),
                "Predicted_Cost": round(float(cost), 2),
                "Readmission_Risk_Probability": round(float(readmission_prob), 3),
                "Readmission_Risk_Level": "High" if readmission_class == 1 else "Low"
            }

            return result

        except Exception as e:
            return {
                "error": str(e)
            }
