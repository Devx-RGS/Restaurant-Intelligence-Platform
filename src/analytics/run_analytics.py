import pandas as pd
from sqlalchemy import create_engine, text
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_CONNECTION_STRING

def run_sql_file(engine, sql_file_path):
    """Reads a .sql file and executes it against PostgreSQL. Returns a DataFrame."""
    with open(sql_file_path, 'r') as file:
        sql_query = file.read()
    
    df_result = pd.read_sql(text(sql_query), con=engine)
    return df_result

def main():
    print("=" * 60)
    print("  ZOMATO ANALYTICS ENGINE")
    print("  Running all SQL queries...")
    print("=" * 60)
    
    engine = create_engine(DB_CONNECTION_STRING)
    
    output_folder = 'data/analytics'
    os.makedirs(output_folder, exist_ok=True)
    
    queries = [
        ("Market Gaps", "src/analytics/market_gaps.sql", "market_gaps.csv"),
        ("Pricing Intelligence", "src/analytics/pricing_intelligence.sql", "pricing_intelligence.csv"),
        ("Competition Density", "src/analytics/competition_density.sql", "competition_density.csv"),
    ]
    
    for query_name, sql_path, csv_name in queries:
        print(f"\nRunning: {query_name}...")
        
        df = run_sql_file(engine, sql_path)
        
        output_path = os.path.join(output_folder, csv_name)
        df.to_csv(output_path, index=False)
        
        print(f"  Rows returned: {df.shape[0]}")
        print(f"  Saved to: {output_path}")
        print(f"  Preview:")
        print(df.head(3).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("  ALL QUERIES COMPLETE!")
    print(f"  Results saved in: {output_folder}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
