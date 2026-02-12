import joblib
import pandas as pd

# Load full pipelines (already contain preprocessor)
los_pipeline = joblib.load("artifacts/los_model.pkl")
cost_pipeline = joblib.load("artifacts/cost_model.pkl")
readmission_pipeline = joblib.load("artifacts/readmission_model.pkl")


def predict_pipeline(input_data: dict):

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Predict
    predicted_los = los_pipeline.predict(df)[0]
    predicted_cost = cost_pipeline.predict(df)[0]
    readmission_prob = readmission_pipeline.predict_proba(df)[0][1]

    # Risk level logic
    if readmission_prob < 0.3:
        risk_level = "Low"
    elif readmission_prob < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    result = {
        "Predicted_Length_of_Stay": float(predicted_los),
        "Predicted_Cost": float(predicted_cost),
        "Readmission_Probability": float(readmission_prob),
        "Risk_Level": risk_level
    }

    return result
