import pandas as pd
import numpy as np

# Load business dataset
def load_data():
    df = pd.read_csv("data/raw/business_transactions.csv")
    return df


# Calculate overall business performance
def business_summary(df):

    summary = {
        "Total Revenue": df["Net_Revenue"].sum(),
        "Total Profit": df["Profit"].sum(),
        "Total Costs": df["Total_Cost"].sum(),
        "Total Discounts": df["Discount_Amount"].sum(),
        "Total Returns": df["Returned"].sum(),
        "Average Profit Margin": df["Profit_Margin"].mean()
    }

    return summary


# Product profitability analysis
def product_analysis(df):

    result = df.groupby("Product").agg(
        Revenue=("Net_Revenue", "sum"),
        Profit=("Profit", "sum"),
        Quantity=("Quantity", "sum"),
        Average_Discount=("Discount_Rate", "mean")
    ).reset_index()

    result["Profit_Margin"] = (
        result["Profit"] / result["Revenue"].replace(0, np.nan)
    ) * 100

    result = result.sort_values("Profit")

    return result


# Customer segment analysis
def customer_analysis(df):

    result = df.groupby("Customer_Segment").agg(
        Revenue=("Net_Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "count"),
        Returns=("Returned", "sum")
    ).reset_index()

    return result


# Regional analysis
def regional_analysis(df):

    result = df.groupby("Region").agg(
        Revenue=("Net_Revenue", "sum"),
        Profit=("Profit", "sum"),
        Orders=("Order_ID", "count")
    ).reset_index()

    return result


# Identify loss-making transactions
def loss_analysis(df):

    losses = df[df["Profit"] < 0].copy()

    return losses.sort_values("Profit")


# Identify excessive discounts
def discount_analysis(df):

    result = df[df["Discount_Rate"] >= 0.20].copy()

    return result.sort_values(
        "Discount_Amount",
        ascending=False
    )


# Execute analytics
if __name__ == "__main__":

    df = load_data()

    print("\n========== PROFITGUARD AI ==========")

    print("\nBUSINESS SUMMARY")
    for key, value in business_summary(df).items():
        print(f"{key}: {value:,.2f}")

    print("\nPRODUCT PROFITABILITY")
    print(product_analysis(df).to_string(index=False))

    print("\nCUSTOMER ANALYSIS")
    print(customer_analysis(df).to_string(index=False))

    print("\nREGIONAL ANALYSIS")
    print(regional_analysis(df).to_string(index=False))

    print("\nLOSS-MAKING TRANSACTIONS")
    print(loss_analysis(df).head(10).to_string(index=False))

    print("\nHIGH DISCOUNT TRANSACTIONS")
    print(discount_analysis(df).head(10).to_string(index=False))

    print("\n========== ANALYSIS COMPLETED ==========")