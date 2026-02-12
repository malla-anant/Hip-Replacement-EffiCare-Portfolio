import joblib
import pandas as pd

los_model = joblib.load("artifacts/los_model.pkl")

model = los_model.named_steps["model"]
preprocessor = los_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

importance = model.feature_importances_

df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values(by="importance", ascending=False)

print(df.head(20))
