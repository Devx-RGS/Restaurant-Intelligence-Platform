import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import joblib
import os
import time

def train_catboost():
    print("Loading ML-ready data (with NLP & Dish features)...", flush=True)
    df = pd.read_csv('data/ml/ml_ready_zomato.csv')
    
    with open('data/ml/categorical_columns.txt', 'r') as f:
        cat_cols = f.read().strip().split('\n')
    
    X = df.drop(columns=['rate'])
    y = df['rate']
    
    for col in cat_cols:
        X[col] = X[col].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data: {X_train.shape[0]} rows", flush=True)
    print(f"Total features: {X_train.shape[1]}", flush=True)
    
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)
    
    # Use the best parameters we found from our earlier massive tuning run
    best_params = {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 3, 'iterations': 1000}
    
    print("\nTraining final CatBoost model...", flush=True)
    start_time = time.time()
    
    model = CatBoostRegressor(
        **best_params,
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
    
    print(f"\nFINAL TEST RESULTS (With NLP & Signature Dish features):", flush=True)
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
    model.save_model('models/zomato_rating_model.cbm')
    joblib.dump(list(X.columns), 'models/model_columns.pkl')
    joblib.dump(cat_cols, 'models/categorical_columns.pkl')
    
    print("\nTuned CatBoost model saved to 'models/' folder successfully!", flush=True)

if __name__ == "__main__":
    train_catboost()
