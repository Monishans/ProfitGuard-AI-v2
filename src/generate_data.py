import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 3000

categories = ["Electronics", "Furniture", "Clothing", "Groceries", "Accessories"]

products = {
    "Electronics": ["Laptop", "Mobile", "Headphones", "Monitor"],
    "Furniture": ["Chair", "Table", "Sofa", "Desk"],
    "Clothing": ["Shirt", "Jeans", "Jacket", "Shoes"],
    "Groceries": ["Rice", "Oil", "Snacks", "Beverages"],
    "Accessories": ["Watch", "Bag", "Wallet", "Belt"]
}

category = np.random.choice(categories, n)
region = np.random.choice(["North", "South", "East", "West"], n)
segment = np.random.choice(
    ["Consumer", "Corporate", "Small Business"], n
)

product = [np.random.choice(products[c]) for c in category]

quantity = np.random.randint(1, 11, n)

unit_price = np.random.uniform(100, 5000, n).round(2)

unit_cost = (
    unit_price * np.random.uniform(0.45, 0.80, n)
).round(2)

discount = np.random.choice(
    [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
    n,
    p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05]
)

gross_sales = quantity * unit_price

discount_amount = gross_sales * discount

sales = gross_sales - discount_amount

product_cost = quantity * unit_cost

shipping_cost = np.random.uniform(20, 300, n)

operating_expense = np.random.uniform(10, 200, n)

return_probability = np.where(discount > 0.25, 0.22, 0.07)

return_probability += np.where(category == "Electronics", 0.04, 0)

returned = np.random.binomial(1, return_probability)

return_amount = sales * returned

return_handling_cost = returned * np.random.uniform(10, 100, n)

unusual = np.random.binomial(1, 0.025, n)

operating_expense += unusual * np.random.uniform(500, 1500, n)

net_revenue = sales - return_amount

total_cost = (
    product_cost
    + shipping_cost
    + operating_expense
    + return_handling_cost
)

profit = net_revenue - total_cost

profit_margin = np.where(
    net_revenue > 0,
    (profit / np.maximum(net_revenue, 1)) * 100,
    0
)

dates = pd.date_range(
    start="2025-01-01",
    periods=n,
    freq="h"
)

df = pd.DataFrame({
    "Order_ID": range(1001, 1001 + n),
    "Order_Date": dates,
    "Product": product,
    "Category": category,
    "Region": region,
    "Customer_Segment": segment,
    "Quantity": quantity,
    "Unit_Price": unit_price,
    "Unit_Cost": unit_cost,
    "Gross_Sales": gross_sales.round(2),
    "Discount_Rate": discount,
    "Discount_Amount": discount_amount.round(2),
    "Sales": sales.round(2),
    "Product_Cost": product_cost.round(2),
    "Shipping_Cost": shipping_cost.round(2),
    "Operating_Expense": operating_expense.round(2),
    "Returned": returned,
    "Return_Amount": return_amount.round(2),
    "Return_Handling_Cost": return_handling_cost.round(2),
    "Net_Revenue": net_revenue.round(2),
    "Total_Cost": total_cost.round(2),
    "Profit": profit.round(2),
    "Profit_Margin": profit_margin.round(2)
})

output = Path("data/raw/business_transactions.csv")
output.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output, index=False)

print("\n===================================")
print(" PROFITGUARD AI - DATASET CREATED")
print("===================================")

print("\nDataset shape:", df.shape)

print("\nFirst 5 transactions:")
print(df.head())

print("\nTotal Revenue:", round(df["Net_Revenue"].sum(), 2))

print("Total Profit:", round(df["Profit"].sum(), 2))

print(
    "Average Profit Margin:",
    round(df["Profit_Margin"].mean(), 2), "%"
)

print("\nReturned Orders:", df["Returned"].sum())

print("\nSaved successfully:", output)