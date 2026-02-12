import os
import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from preprocessing import load_and_prepare_data


(
    X_train, X_test,
    _, _,
    y_cost_train, y_cost_test,
    _, _,
    preprocessor
) = load_and_prepare_data()


# 🔥 Replace with tuned parameters from notebook
cost_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", cost_model)
    ]
)

pipeline.fit(X_train, y_cost_train)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(pipeline, "artifacts/cost_model.pkl")

print("✅ Cost model trained and saved.")

from sklearn.metrics import r2_score, mean_absolute_error

preds = pipeline.predict(X_test)

print("R2 Score:", r2_score(y_cost_test, preds))
print("MAE:", mean_absolute_error(y_cost_test, preds))
