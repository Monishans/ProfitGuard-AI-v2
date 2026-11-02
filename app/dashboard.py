
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ProfitGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ ProfitGuard AI")
st.subheader("AI-Powered Hidden Profit Leakage Detector")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Dashboard",
        "Profit Leakage",
        "AI Anomaly Detection",
        "Customer Intelligence",
        "Profit Prediction",
        "What-if Simulator",
        "AI Recommendations"
    ]
)

uploaded = st.sidebar.file_uploader("Upload business CSV", type=["csv"])

@st.cache_data
def load_default():
    return pd.read_csv("data/raw/business_transactions.csv")

df = pd.read_csv(uploaded) if uploaded else load_default()

required = [
    "Net_Revenue", "Profit", "Total_Cost",
    "Discount_Rate", "Discount_Amount",
    "Returned", "Operating_Expense",
    "Shipping_Cost", "Product", "Category",
    "Region", "Customer_Segment", "Quantity",
    "Profit_Margin", "Order_ID"
]

if not all(c in df.columns for c in required):
    st.error("Uploaded CSV does not contain all required columns.")
    st.stop()

st.sidebar.success(f"{len(df):,} transactions loaded")

revenue = df["Net_Revenue"].sum()
profit = df["Profit"].sum()
cost = df["Total_Cost"].sum()
discounts = df["Discount_Amount"].sum()
margin = profit / revenue * 100 if revenue else 0
losses = df[df["Profit"] < 0]

if page == "Executive Dashboard":

    st.header("Executive Business Overview")

    a, b, c, d = st.columns(4)
    a.metric("Net Revenue", f"₹{revenue:,.0f}")
    b.metric("Total Profit", f"₹{profit:,.0f}")
    c.metric("Profit Margin", f"{margin:.2f}%")
    d.metric("Loss Transactions", f"{len(losses):,}")

    e, f, g = st.columns(3)
    e.metric("Total Costs", f"₹{cost:,.0f}")
    f.metric("Discount Leakage", f"₹{discounts:,.0f}")
    g.metric("Returned Orders", f"{df.Returned.sum():,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Category")
        chart = df.groupby("Category", as_index=False)["Net_Revenue"].sum()
        st.plotly_chart(
            px.bar(chart, x="Category", y="Net_Revenue",
                   title="Net Revenue by Category"),
            use_container_width=True
        )

    with col2:
        st.subheader("Profit by Region")
        chart = df.groupby("Region", as_index=False)["Profit"].sum()
        st.plotly_chart(
            px.bar(chart, x="Region", y="Profit",
                   color="Profit", title="Regional Profit"),
            use_container_width=True
        )

    st.subheader("Profit Distribution")
    st.plotly_chart(
        px.histogram(df, x="Profit", nbins=40),
        use_container_width=True
    )

    st.subheader("Recent Transactions")
    st.dataframe(df.head(20), use_container_width=True)

elif page == "Profit Leakage":

    st.header("Profit Leakage Investigation")

    st.metric("Total Loss-Making Transactions", len(losses))
    st.metric("Total Loss Amount", f"₹{abs(losses.Profit.sum()):,.2f}")

    product = df.groupby("Product", as_index=False).agg(
        Revenue=("Net_Revenue", "sum"),
        Profit=("Profit", "sum"),
        Discount=("Discount_Amount", "sum"),
        Orders=("Order_ID", "count")
    )

    product["Margin"] = (
        product["Profit"] /
        product["Revenue"].replace(0, np.nan) * 100
    )

    st.subheader("Product Profitability")
    st.dataframe(
        product.sort_values("Profit"),
        use_container_width=True
    )

    st.plotly_chart(
        px.bar(product.sort_values("Profit"),
               x="Product", y="Profit",
               color="Profit", title="Product-wise Profit"),
        use_container_width=True
    )

    st.subheader("High Discount Transactions")
    st.dataframe(
        df[df.Discount_Rate >= 0.20].sort_values(
            "Discount_Amount", ascending=False
        ).head(30),
        use_container_width=True
    )

    st.subheader("Loss-Making Transactions")
    st.dataframe(losses.head(50), use_container_width=True)

    st.subheader("Regional Profitability")
    region = df.groupby("Region", as_index=False)["Profit"].sum()
    st.plotly_chart(
        px.bar(region, x="Region", y="Profit", color="Profit"),
        use_container_width=True
    )

elif page == "AI Anomaly Detection":

    st.header("AI Anomaly Detection")
    st.write("Isolation Forest identifies unusual business transactions.")

    features = [
        "Discount_Rate",
        "Operating_Expense",
        "Shipping_Cost",
        "Profit_Margin",
        "Return_Amount",
        "Quantity"
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    df["Anomaly"] = model.fit_predict(X)
    anomalies = df[df.Anomaly == -1]

    a, b = st.columns(2)
    a.metric("Anomalies Detected", len(anomalies))
    b.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%")

    st.subheader("Unusual Transactions")
    st.dataframe(
        anomalies.sort_values("Operating_Expense", ascending=False),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            df, x="Discount_Rate", y="Profit",
            color=df["Anomaly"].map({1: "Normal", -1: "Anomaly"}),
            title="Anomaly Detection: Discount vs Profit"
        ),
        use_container_width=True
    )

elif page == "Customer Intelligence":

    st.header("Customer Intelligence")

    segment = df.groupby("Customer_Segment", as_index=False).agg(
        Revenue=("Net_Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "count"),
        Returns=("Returned", "sum"),
        Discount=("Discount_Amount", "sum")
    )

    st.subheader("Customer Segment Performance")
    st.dataframe(segment, use_container_width=True)

    st.plotly_chart(
        px.bar(segment, x="Customer_Segment", y="Profit",
               color="Profit"),
        use_container_width=True
    )

    st.subheader("Customer Behaviour Clustering")

    features = [
        "Net_Revenue", "Discount_Rate",
        "Profit", "Total_Cost", "Quantity"
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Customer_Cluster"] = kmeans.fit_predict(scaled)

    clusters = df.groupby("Customer_Cluster", as_index=False).agg(
        Revenue=("Net_Revenue", "mean"),
        Profit=("Profit", "mean"),
        Discount=("Discount_Rate", "mean"),
        Transactions=("Order_ID", "count")
    )

    st.dataframe(clusters, use_container_width=True)

    st.plotly_chart(
        px.scatter(
            df, x="Net_Revenue", y="Profit",
            color=df["Customer_Cluster"].astype(str),
            title="Transaction Clusters"
        ),
        use_container_width=True
    )

elif page == "Profit Prediction":

    st.header("Machine Learning: Profit Prediction")

    features = [
        "Net_Revenue", "Total_Cost",
        "Discount_Rate", "Shipping_Cost",
        "Operating_Expense", "Quantity",
        "Returned"
    ]

    X = df[features].fillna(0)
    y = df["Profit"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)
    predictions = model.predict(X)

    mae = np.mean(np.abs(y - predictions))
    rmse = np.sqrt(np.mean((y - predictions) ** 2))

    a, b = st.columns(2)
    a.metric("Mean Absolute Error", f"₹{mae:,.2f}")
    b.metric("Root Mean Squared Error", f"₹{rmse:,.2f}")

    result = pd.DataFrame({
        "Actual Profit": y,
        "Predicted Profit": predictions
    })

    st.plotly_chart(
        px.scatter(
            result, x="Actual Profit", y="Predicted Profit",
            title="Actual vs Predicted Profit"
        ),
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.plotly_chart(
        px.bar(importance, x="Feature", y="Importance"),
        use_container_width=True
    )

elif page == "What-if Simulator":

    st.header("Profit Recovery Simulator")

    st.write("Simulate changes to discounts and operating expenses.")

    discount_reduction = st.slider(
        "Reduce existing discounts by (%)",
        0, 50, 10
    )

    expense_reduction = st.slider(
        "Reduce operating expenses by (%)",
        0, 50, 10
    )

    price_change = st.slider(
        "Increase selling price by (%)",
        0, 20, 0
    )

    baseline = profit

    simulated = (
        df["Net_Revenue"]
        + df["Discount_Amount"] * discount_reduction / 100
        + df["Net_Revenue"] * price_change / 100
        - df["Operating_Expense"] * expense_reduction / 100
        - df["Product_Cost"] * 0
    ).sum() - (
        df["Total_Cost"].sum()
        - df["Operating_Expense"].sum()
    )

    improvement = simulated - baseline

    a, b, c = st.columns(3)
    a.metric("Current Profit", f"₹{baseline:,.2f}")
    b.metric("Simulated Profit", f"₹{simulated:,.2f}")
    c.metric("Estimated Improvement", f"₹{improvement:,.2f}")

    st.info(
        "This is a simplified scenario estimate. "
        "It assumes sales volume and customer demand remain unchanged."
    )

elif page == "AI Recommendations":

    st.header("AI Business Recommendations")

    recommendations = []

    if margin < 15:
        recommendations.append(
            "Profit margin is below 15%. Review costs and pricing."
        )

    if len(losses) > 0:
        recommendations.append(
            f"{len(losses)} transactions are loss-making. Investigate these orders."
        )

    high_discount = df[df.Discount_Rate >= 0.20]

    if len(high_discount) > 0:
        recommendations.append(
            f"{len(high_discount)} transactions have discounts of 20% or more. Review discount policies."
        )

    if df.Returned.mean() > 0.10:
        recommendations.append(
            "Return rate exceeds 10%. Investigate product quality and customer complaints."
        )

    product_profit = df.groupby("Product")["Profit"].sum()

    for name, value in product_profit.items():
        if value < 0:
            recommendations.append(
                f"Product {name} has negative total profit. Review its pricing and costs."
            )

    expensive = df[df.Operating_Expense > df.Operating_Expense.quantile(0.95)]

    if len(expensive) > 0:
        recommendations.append(
            f"{len(expensive)} transactions have unusually high operating expenses."
        )

    if not recommendations:
        recommendations.append(
            "No major rule-based leakage warning was detected. Continue monitoring."
        )

    for i, recommendation in enumerate(recommendations, 1):
        st.warning(f"{i}. {recommendation}")

    st.subheader("Download Business Report")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Analyzed Dataset",
        csv,
        "profitguard_analysis.csv",
        "text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("ProfitGuard AI | Business Analytics")
