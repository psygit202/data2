import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(n=2000):
    np.random.seed(42)
    
    categories = ["Electronics", "Clothing", "Beauty", "Groceries", "Home Decor"]
    payment_methods = ["Credit Card", "Cash", "Digital Wallet", "Debit Card"]
    cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier"]
    genders = ["Male", "Female", "Other"]
    
    start_date = datetime(2025, 1, 1)
    
    data = {
        "Transaction ID": [f"T{100000+i}" for i in range(n)],
        "Customer ID": [f"C{1000+np.random.randint(0, 500)}" for i in range(n)],
        "Date": [start_date + timedelta(days=np.random.randint(0, 365), hours=np.random.randint(0, 24)) for i in range(n)],
        "Product Category": np.random.choice(categories, n),
        "Quantity": np.random.randint(1, 10, n),
        "Price Per Unit": np.random.uniform(5, 500, n).round(2),
        "Payment Method": np.random.choice(payment_methods, n),
        "Store Location": np.random.choice(cities, n),
        "Discount Applied": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "Age": np.random.randint(18, 75, n),
        "Gender": np.random.choice(genders, n)
    }
    
    df = pd.DataFrame(data)
    df["Total Amount"] = (df["Quantity"] * df["Price Per Unit"]).round(2)
    
    # Sort by date for realism
    df = df.sort_values("Date").reset_index(drop=True)
    
    df.to_csv("retail_transactional_dataset.csv", index=False)
    print(f"Generated {n} rows of mock data.")

if __name__ == "__main__":
    generate_mock_data()
