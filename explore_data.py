import pandas as pd

print("Loading data...")
df = pd.read_csv('data/raw/zomato.csv')

print(f"\nDATASET SIZE")

print(f"Total Rows (Restaurants): {df.shape[0]}")
print(f"Total Columns (Features): {df.shape[1]}")

# Datatype of columns
dt = df.dtypes
print(dt)

print(f"\n           MISSING VALUES CHECK")
print(df.isnull().sum())