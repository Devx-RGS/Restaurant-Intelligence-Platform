import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import os
import ast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Import our database connection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_CONNECTION_STRING

analyzer = SentimentIntensityAnalyzer()

def process_reviews(review_str):
    if pd.isna(review_str) or review_str == '[]':
        return 0, 0, 0.0
    try:
        reviews = ast.literal_eval(review_str)
        if not isinstance(reviews, list) or len(reviews) == 0:
            return 0, 0, 0.0 # 0 review, 0 length and 0 sentiment score
            
        total_len = 0
        total_sentiment = 0.0
        valid_reviews = 0
        
        for rating_text, review_text in reviews:
            clean_text = review_text.replace('RATED\n  ', '').strip()
            if not clean_text:
                continue
                
            total_len += len(clean_text)
            total_sentiment += analyzer.polarity_scores(clean_text)['compound']
            valid_reviews += 1
            
        if valid_reviews == 0:
            return 0, 0, 0.0
            
        return valid_reviews, total_len / valid_reviews, total_sentiment / valid_reviews
    except:
        return 0, 0, 0.0

def prepare_ml_data():
    print("Connecting to database...")
    engine = create_engine(DB_CONNECTION_STRING)
    
    # Load the clean restaurants table
    df = pd.read_sql("SELECT * FROM restaurants", con=engine)
    print(f"Loaded {df.shape[0]} rows.")
    
    # =========================================================
    # NLP & Review Analysis Report
    # =========================================================
    print("\n" + "="*50)
    print("REVIEWS & NLP ANALYSIS REPORT")
    print("="*50)
    
    # Calculate missing review statistics
    is_empty = df['reviews_list'].isna() | (df['reviews_list'] == '[]')
    total_null = is_empty.sum()
    pct_null = (total_null / len(df)) * 100
    non_empty = len(df) - total_null
    
    print(f"Total rows with NO reviews: {total_null}")
    print(f"Percentage empty: {pct_null:.1f}%")
    print(f"Number of non-empty reviews to process: {non_empty}")
    
    print("\nProcessing NLP sentiment analysis...")
    review_features = df['reviews_list'].apply(process_reviews)
    df['review_count'] = [r[0] for r in review_features]
    df['avg_review_length'] = [r[1] for r in review_features]
    df['avg_sentiment_score'] = [r[2] for r in review_features]
    
    # =========================================================
    # Dish Liked Features
    # =========================================================
    print("\nExtracting signature dish features...")
    df['has_signature_dishes'] = df['dish_liked'].notna().astype(int)
    df['num_liked_dishes'] = df['dish_liked'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    
    # =========================================================
    # Standard Features
    # =========================================================
    print("Processing standard numerical & categorical features...")
    df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
    df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})
    
    df['log_votes'] = np.log1p(df['votes'])
    df['log_cost'] = np.log1p(df['cost_for_two'])
    
    df['votes_per_cost'] = df['votes'] / (df['cost_for_two'] + 1)
    df['cost_x_booktable'] = df['cost_for_two'] * df['book_table']
    
    df['cuisines'] = df['cuisines'].fillna('Unknown')
    df['num_cuisines'] = df['cuisines'].apply(lambda x: len(x.split(',')))
    
    all_cuisines = pd.Series(','.join(df['cuisines']).split(',')).str.strip()
    top_cuisines = all_cuisines.value_counts().head(15).index.tolist()
    for cuisine in top_cuisines:
        col_name = f"cuisine_{cuisine.replace(' ', '_')}"
        df[col_name] = df['cuisines'].str.contains(cuisine, case=False, na=False).astype(int)
        
    name_counts = df['name'].value_counts()
    df['is_chain'] = df['name'].map(name_counts).apply(lambda x: 1 if x >= 3 else 0)
    df['chain_outlets'] = df['name'].map(name_counts)
    
    for col in ['location', 'rest_type', 'listed_in(type)', 'listed_in(city)']:
        df[col] = df[col].fillna('Unknown').astype(str)
        
    # Assemble Final Features
    numerical_features = [
        'online_order', 'book_table', 'cost_for_two', 'votes',
        'log_votes', 'log_cost', 'votes_per_cost', 'cost_x_booktable',
        'num_cuisines', 'is_chain', 'chain_outlets',
        'review_count', 'avg_review_length', 'avg_sentiment_score',
        'has_signature_dishes', 'num_liked_dishes'
    ]
    cuisine_cols = [f"cuisine_{c.replace(' ', '_')}" for c in top_cuisines]
    categorical_cols = ['location', 'rest_type', 'listed_in(type)', 'listed_in(city)']
    
    features_df = df[numerical_features + cuisine_cols + categorical_cols]
    target_df = df['rate']
    
    ml_data = pd.concat([features_df, target_df], axis=1)
    
    os.makedirs('data/ml', exist_ok=True)
    output_path = 'data/ml/ml_ready_zomato.csv'
    ml_data.to_csv(output_path, index=False)
    
    with open('data/ml/categorical_columns.txt', 'w') as f:
        f.write('\n'.join(categorical_cols))
    
    print(f"\nML Preparation complete! Saved to {output_path}")
    print(f"New shape: {ml_data.shape[0]} rows, {ml_data.shape[1]} columns")

if __name__ == "__main__":
    prepare_ml_data()