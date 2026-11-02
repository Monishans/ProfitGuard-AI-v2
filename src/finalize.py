
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA = Path("data/raw/business_transactions.csv")
DB = Path("data/profitguard.db")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA)

# 1. SQL DATABASE
conn = sqlite3.connect(DB)

df.to_sql(
    "transactions",
    conn,
    if_exists="replace",
    index=False
)

summary = df.groupby("Category").agg(
    Revenue=("Net_Revenue", "sum"),
    Profit=("Profit", "sum"),
    Orders=("Order_ID", "count")
).reset_index()

summary.to_sql(
    "category_summary",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("SQL DATABASE CREATED")

# 2. TRAIN PROFIT PREDICTION MODEL
features = [
    "Net_Revenue",
    "Total_Cost",
    "Discount_Rate",
    "Shipping_Cost",
    "Operating_Expense",
    "Quantity",
    "Returned"
]

X = df[features].fillna(0)
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = {
    "MAE": float(mean_absolute_error(y_test, predictions)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
    "R2": float(r2_score(y_test, predictions))
}

joblib.dump(model, MODEL_DIR / "profit_model.joblib")

print("PROFIT MODEL SAVED")
print("MODEL METRICS:", metrics)

# 3. ISOLATION FOREST
anomaly_features = [
    "Discount_Rate",
    "Operating_Expense",
    "Shipping_Cost",
    "Profit_Margin",
    "Return_Amount",
    "Quantity"
]

anomaly_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

df["Anomaly"] = anomaly_model.fit_predict(
    df[anomaly_features].fillna(0)
)

joblib.dump(
    anomaly_model,
    MODEL_DIR / "anomaly_model.joblib"
)

df[df["Anomaly"] == -1].to_csv(
    REPORT_DIR / "anomalies.csv",
    index=False
)

print("ANOMALY MODEL SAVED")

# 4. CUSTOMER SEGMENTATION
cluster_features = [
    "Net_Revenue",
    "Discount_Rate",
    "Profit",
    "Total_Cost",
    "Quantity"
]

scaler = StandardScaler()
scaled = scaler.fit_transform(
    df[cluster_features].fillna(0)
)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Customer_Cluster"] = kmeans.fit_predict(scaled)

joblib.dump(kmeans, MODEL_DIR / "customer_clusters.joblib")
joblib.dump(scaler, MODEL_DIR / "customer_scaler.joblib")

df.to_csv(
    REPORT_DIR / "complete_analysis.csv",
    index=False
)

print("CUSTOMER CLUSTERING SAVED")

# 5. SAVE MODEL EVALUATION
with open(REPORT_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 6. BUSINESS REPORT
report = {
    "Total Transactions": int(len(df)),
    "Total Revenue": float(df["Net_Revenue"].sum()),
    "Total Profit": float(df["Profit"].sum()),
    "Total Costs": float(df["Total_Cost"].sum()),
    "Discount Amount": float(df["Discount_Amount"].sum()),
    "Loss Transactions": int((df["Profit"] < 0).sum()),
    "Returned Orders": int(df["Returned"].sum()),
    "Anomalies": int((df["Anomaly"] == -1).sum()),
    "Average Profit Margin": float(df["Profit_Margin"].mean())
}

with open(REPORT_DIR / "business_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nALL FINAL INTEGRATIONS COMPLETED!")
print("Database:", DB)
print("Models:", MODEL_DIR)
print("Reports:", REPORT_DIR)
