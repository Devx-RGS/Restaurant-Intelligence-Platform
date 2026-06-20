import pandas as pd
import numpy as np

def clean_rating_text(text):
    if pd.isnull(text):
        return np.nan
    
    text = str(text).strip()
    if text == 'NEW' or text == '-':
        return np.nan
        
    text = text.replace('/5', '').strip()
    try:
        return float(text)
    except:
        return np.nan

def clean_zomato_data(file_path):
    print("Starting ETL Process: Transformation")
    
    print("Loading raw data...")
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape[0]} rows")
    
    
    print("Dropping useless columns...")
    columns_to_drop = ['url', 'phone', 'dish_liked', 'menu_item', 'reviews_list']
    df = df.drop(columns=columns_to_drop)
    
    print("Removing duplicate restaurants...")
    df = df.drop_duplicates(subset=['name', 'address'], keep='first')
    
    print("Cleaning the 'rate' column...")
    df['rate'] = df['rate'].apply(clean_rating_text)
    
    df = df.dropna(subset=['rate'])
    
    print("Cleaning the 'cost' column...")
    df = df.rename(columns={'approx_cost(for two people)': 'cost_for_two'})
    
    df['cost_for_two'] = df['cost_for_two'].astype(str).str.replace(',', '')
    
    df['cost_for_two'] = pd.to_numeric(df['cost_for_two'], errors='coerce')
    
    df = df.dropna(subset=['cost_for_two'])
    
    # 2.5 Drop rows where rest_type or cuisines are missing (only ~40 rows)
    print("Dropping rows with missing rest_type or cuisines...")
    df = df.dropna(subset=['rest_type', 'cuisines'])
    
    print(f"Final shape after ALL cleaning: {df.shape[0]} rows")
    
    return df

if __name__ == "__main__":
    import os
    
    df_clean = clean_zomato_data('data/raw/zomato.csv')
    
    print("\nRemaining Columns:", df_clean.columns.tolist())
    
    print("\nSaving cleaned data...")
    os.makedirs('data/cleaned', exist_ok=True)
    
    output_path = 'data/cleaned/cleaned_zomato.csv'
    df_clean.to_csv(output_path, index=False)
    
    print(f"Success! Cleaned data saved to: {output_path}")
