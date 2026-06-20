import pandas as pd
from sqlalchemy import create_engine
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_CONNECTION_STRING

def load_data_to_postgres(csv_path, table_name):
    print(f"Starting ETL Process: LOAD")
    
    print(f"Loading {csv_path} into memory...")
    df = pd.read_csv(csv_path)
    
    print("Connecting to PostgreSQL database 'zomato_db'...")
    engine = create_engine(DB_CONNECTION_STRING)
    
   
    print(f"Pushing {df.shape[0]} rows into the '{table_name}' table.")
    try:
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print("Data successfully loaded into PostgreSQL!")
    except Exception as e:
        print(" ERROR: Could not load data.")
        print(f"Details: {e}")

if __name__ == "__main__":
    load_data_to_postgres('data/cleaned/cleaned_zomato.csv', 'restaurants')
