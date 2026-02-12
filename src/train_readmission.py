import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from preprocessing import load_and_prepare_data


(
    X_train, X_test,
    _, _,
    _, _,
    y_read_train, y_read_test,
    preprocessor
) = load_and_prepare_data()


# Balanced ExtraTrees Classifier
read_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", read_model)
    ]
)

pipeline.fit(X_train, y_read_train)

# Evaluate
preds = pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_read_test, preds))

# Save Model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipeline, "artifacts/readmission_model.pkl")

print("\n✅ Readmission model trained and saved.")
