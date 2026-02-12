from inference_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

print("\nExpected Input Features:\n")
print(pipeline.expected_features)
print("\nTotal Features:", len(pipeline.expected_features))
