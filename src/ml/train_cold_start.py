import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import joblib
import os
import time

def train_cold_start_model():
    print("Loading ML-ready data for Cold Start Model...", flush=True)
    df = pd.read_csv('data/ml/ml_ready_zomato.csv')
    
    with open('data/ml/categorical_columns.txt', 'r') as f:
        cat_cols = f.read().strip().split('\n')
        
    # Exclude all post-launch features (votes, reviews, sentiment, etc.)
    post_launch_features = [
        'votes', 'log_votes', 'votes_per_cost', 
        'review_count', 'avg_review_length', 'avg_sentiment_score',
        'has_signature_dishes', 'num_liked_dishes'
    ]
    
    X = df.drop(columns=['rate'] + post_launch_features)
    y = df['rate']
    
    for col in cat_cols:
        X[col] = X[col].astype(str)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data: {X_train.shape[0]} rows", flush=True)
    print(f"Total features used (excluding post-launch): {X_train.shape[1]}", flush=True)
    
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)
    
    # We will use slightly gentler parameters for the cold start to avoid overfitting on limited data
    params = {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 5, 'iterations': 1000}
    
    print("\nTraining Cold Start CatBoost model...", flush=True)
    start_time = time.time()
    
    model = CatBoostRegressor(
        **params,
        eval_metric='R2', 
        random_seed=42, 
        verbose=100,
        early_stopping_rounds=50
    )
    
    model.fit(train_pool, eval_set=test_pool)
    
    elapsed = time.time() - start_time
    print(f"\nSUCCESS: Training complete! Took {elapsed:.1f} seconds", flush=True)
    
    preds = model.predict(test_pool)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    
    print(f"\nFINAL TEST RESULTS (Cold Start Model):", flush=True)
    print(f"  R2 Score: {r2:.4f}", flush=True)
    print(f"  MSE:      {mse:.4f}", flush=True)
    print(f"  RMSE:     {np.sqrt(mse):.4f}", flush=True)
    
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    print(f"\nTOP 15 MOST IMPORTANT FEATURES:", flush=True)
    for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
        print(f"  {i}. {feat}: {imp:.4f}", flush=True)
        
    os.makedirs('models', exist_ok=True)
    model.save_model('models/zomato_cold_start_model.cbm')
    joblib.dump(list(X.columns), 'models/cold_start_columns.pkl')
    
    print("\nCold Start Model saved to 'models/' folder successfully!", flush=True)

if __name__ == "__main__":
    train_cold_start_model()
