import joblib
from preprocessing import load_and_prepare_data
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import numpy as np

(
    X_train, X_test,
    y_los_train, y_los_test,
    y_cost_train, y_cost_test,
    y_read_train, y_read_test,
    _
) = load_and_prepare_data()

los_model = joblib.load("artifacts/los_model.pkl")
cost_model = joblib.load("artifacts/cost_model.pkl")
read_model = joblib.load("artifacts/readmission_model.pkl")

# ---------------- LOS ----------------
los_pred = los_model.predict(X_test)
print("\nLOS Model")
print("R2:", r2_score(y_los_test, los_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_los_test, los_pred)))

# ---------------- COST ----------------
cost_pred = cost_model.predict(X_test)
print("\nCost Model")
print("R2:", r2_score(y_cost_test, cost_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_cost_test, cost_pred)))

# ---------------- READMISSION ----------------
read_pred = read_model.predict(X_test)
print("\nReadmission Model")
print("Accuracy:", accuracy_score(y_read_test, read_pred))
print(classification_report(y_read_test, read_pred))
