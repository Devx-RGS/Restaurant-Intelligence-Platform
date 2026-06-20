import pandas as pd

df = pd.read_csv(r'data\raw\zomato.csv', encoding='latin-1')

empty_menu = df[df['menu_item'] == '[]'].shape[0]
total = df.shape[0]

print(f"Total rows: {total}")
print(f"Empty menu_item ([]): {empty_menu}")
print(f"Percentage: {round(empty_menu/total*100, 2)}%")