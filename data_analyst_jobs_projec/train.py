import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import joblib

# Load dataset
df = pd.read_csv("DataAnalyst.csv")

# --- Data Cleaning ---
df = df.drop(df.columns[0], axis=1)
df["Salary Estimate"] = df["Salary Estimate"].replace("-1", np.nan)

# Extract salaries
df["MinSalary"] = df["Salary Estimate"].str.extract(r"\$(\d+)K").astype(float)
df["MaxSalary"] = df["Salary Estimate"].str.extract(r"-\$(\d+)K").astype(float)
df["AvgSalary"] = (df["MinSalary"] + df["MaxSalary"]) / 2

# Drop missing salaries
df = df.dropna(subset=["AvgSalary"])

# Fix founded
df["Founded"] = df["Founded"].replace(-1, np.nan)
df["Founded"].fillna(df["Founded"].median(), inplace=True)

# Extract skills
df["Python"] = df["Job Description"].str.contains("Python", case=False, na=False).astype(int)
df["Excel"] = df["Job Description"].str.contains("Excel", case=False, na=False).astype(int)
df["SQL"]   = df["Job Description"].str.contains("SQL", case=False, na=False).astype(int)
df["Tech_Skills"] = df["Python"] + df["Excel"] + df["SQL"]

# Remove outliers
q1, q99 = df["AvgSalary"].quantile([0.01, 0.99])
df = df[(df["AvgSalary"] >= q1) & (df["AvgSalary"] <= q99)]

# Features and target
X = df[["Rating", "Tech_Skills", "Founded"]]
y = df["AvgSalary"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "salary_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and Scaler saved!")
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

