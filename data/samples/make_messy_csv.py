"""Generates data/samples/messy_sales.csv, a deliberately dirty dataset for testing.

Run:  python data/samples/make_messy_csv.py
Problems injected: duplicate rows, mixed date formats, inconsistent category spellings,
numbers stored as text with commas, '-' placeholders, missing values, and outliers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 400

region = rng.choice(["West", "North", "South", "East"], n, p=[0.4, 0.25, 0.2, 0.15])
category = rng.choice(["Furniture", "Technology", "Office Supplies"], n)
discount = rng.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3], n)
quantity = rng.integers(1, 10, n).astype(float)
unit_price = rng.uniform(100, 2000, n)
sales = quantity * unit_price * (1 - discount)
profit = sales * (0.35 - 1.2 * discount) + rng.normal(0, 40, n)
age = rng.integers(18, 70, n).astype(float)


def mess_text(v):
    x = rng.random()
    if x < 0.12:
        return v.lower()
    if x < 0.20:
        return f" {v} "
    if x < 0.25:
        return v.upper()
    return v


dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
fmts = ["%d/%m/%Y", "%Y-%m-%d", "%d %b %Y"]
date_str = [d.strftime(fmts[rng.integers(0, 3)]) for d in dates]

df = pd.DataFrame({
    "OrderID": np.arange(1001, 1001 + n),
    "OrderDate": date_str,
    "Region": [mess_text(r) for r in region],
    "Category": [mess_text(c) if c == "Office Supplies" else c for c in category],
    "Quantity": quantity,
    "Discount": discount,
    "Sales": [f"{v:,.2f}" for v in sales],
    "Profit": profit,
    "CustomerAge": age,
})

# missing values and placeholders
df.loc[rng.choice(n, 30, replace=False), "Quantity"] = np.nan
df.loc[rng.choice(n, 55, replace=False), "CustomerAge"] = np.nan
df.loc[rng.choice(n, 8, replace=False), "Sales"] = "-"
df.loc[rng.choice(n, 4, replace=False), "Sales"] = "--"
df.loc[rng.choice(n, 8, replace=False), "Region"] = np.nan
# outliers
df.loc[rng.choice(n, 6, replace=False), "Profit"] *= 12
# exact duplicate rows
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)

out = Path(__file__).with_name("messy_sales.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows")
