import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from preprocessing import load_and_prepare_data


(
    X_train, X_test,
    y_los_train, y_los_test,
    _, _,
    _, _,
    preprocessor
) = load_and_prepare_data()


# 🔥 Replace with exact tuned parameters from your notebook
los_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    min_samples_split=5,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", los_model)
    ]
)

pipeline.fit(X_train, y_los_train)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipeline, "artifacts/los_model.pkl")

print("✅ LOS model trained and saved.")

from sklearn.metrics import r2_score, mean_absolute_error

preds = pipeline.predict(X_test)

print("R2 Score:", r2_score(y_los_test, preds))
print("MAE:", mean_absolute_error(y_los_test, preds))
