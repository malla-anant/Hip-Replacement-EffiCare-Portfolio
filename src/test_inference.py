import pandas as pd
from inference_pipeline import PredictionPipeline


# --------------------------------------------------
# Load Original Dataset to Get Sample Row
# --------------------------------------------------

df = pd.read_csv("data/hip_replacement.csv")

# Remove columns not used during training
columns_to_remove = [
    "length_of_stay",
    "total_costs",
    "readmission",
    "patient_disposition",
    "operating_certificate_number",
    "facility_id",
    "facility_name",
    "attending_provider_license_number",
    "operating_provider_license_number"
]

df = df.drop(columns=columns_to_remove, errors="ignore")

# Take one real patient record
sample_input = df.iloc[0].to_dict()

# --------------------------------------------------
# Load Pipeline
# --------------------------------------------------

pipeline = PredictionPipeline()

# --------------------------------------------------
# Run Prediction
# --------------------------------------------------

result = pipeline.predict(sample_input)

print("\nSample Input:")
print(sample_input)

print("\nPrediction Result:")
print(result)
