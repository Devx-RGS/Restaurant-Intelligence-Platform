import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import sys

class ZomatoPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.analyzer = SentimentIntensityAnalyzer()
        
        print("Loading prediction engines...")
        
        # Load Mode A Model (Cold Start)
        self.cold_model = CatBoostRegressor()
        self.cold_model.load_model(os.path.join(models_dir, 'zomato_cold_start_model.cbm'))
        self.cold_cols = joblib.load(os.path.join(models_dir, 'cold_start_columns.pkl'))
        
        # Load Mode B Model (Full Optimization)
        self.opt_model = CatBoostRegressor()
        self.opt_model.load_model(os.path.join(models_dir, 'zomato_rating_model.cbm'))
        self.opt_cols = joblib.load(os.path.join(models_dir, 'model_columns.pkl'))
        
        # Load Categorical columns definition
        self.cat_cols = joblib.load(os.path.join(models_dir, 'categorical_columns.pkl'))
        
        # Define the exact 15 cuisines the model expects
        self.top_cuisines = [
            'North Indian', 'Chinese', 'South Indian', 'Fast Food', 'Desserts',
            'Biryani', 'Bakery', 'Beverages', 'Cafe', 'Continental', 
            'Ice Cream', 'Street Food', 'Italian', 'Mithai', 'Burger'
        ]

    def _base_feature_engineering(self, params, target_cols):
        # Create an empty dictionary with all required columns set to 0
        features = {col: 0 for col in target_cols}
        
        # 1. Fill basic numericals
        features['online_order'] = 1 if params.get('online_order') else 0
        features['book_table'] = 1 if params.get('book_table') else 0
        features['cost_for_two'] = float(params.get('cost_for_two', 500))
        features['is_chain'] = 1 if params.get('is_chain') else 0
        features['chain_outlets'] = int(params.get('chain_outlets', 0))
        
        # 2. Mathematical features
        features['log_cost'] = np.log1p(features['cost_for_two'])
        features['cost_x_booktable'] = features['cost_for_two'] * features['book_table']
        
        # 3. Cuisine features
        cuisines_list = params.get('cuisines', [])
        features['num_cuisines'] = len(cuisines_list) if cuisines_list else 1
        
        # Check against top 15 cuisines
        for cuisine in self.top_cuisines:
            col_name = f"cuisine_{cuisine.replace(' ', '_')}"
            if col_name in features:
                features[col_name] = 1 if cuisine in cuisines_list else 0
        
        # 4. Fill categoricals (Native CatBoost)
        for cat in self.cat_cols:
            features[cat] = str(params.get(cat, 'Unknown'))
            
        return features

    def simulate_new_restaurant(self, **params):
        """
        MODE A: The Cold Start Predictor
        Uses only market features to output the Day-1 Baseline Rating.
        """
        features = self._base_feature_engineering(params, self.cold_cols)
        
        df_input = pd.DataFrame([features])
        prediction = self.cold_model.predict(df_input)[0]
        
        return prediction

    def optimize_existing_restaurant(self, pasted_reviews_text="", **params):
        """
        MODE B: The Existing Restaurant Optimizer
        Uses full features + live NLP on pasted reviews text.
        """
        features = self._base_feature_engineering(params, self.opt_cols)
        
        # Calculate derived numerical features based on post-launch metrics
        features['votes'] = int(params.get('votes', 0))
        features['log_votes'] = np.log1p(features['votes'])
        features['votes_per_cost'] = features['votes'] / (features['cost_for_two'] + 1)
        
        # Determine signature dish presence
        features['num_liked_dishes'] = int(params.get('num_liked_dishes', 0))
        features['has_signature_dishes'] = 1 if features['num_liked_dishes'] > 0 else 0
        
        # Process and score raw review text using VADER NLP
        valid_reviews = 0
        total_len = 0
        total_sentiment = 0.0
        
        if pasted_reviews_text.strip():
            # Split pasted text by line breaks
            reviews = pasted_reviews_text.strip().split('\n')
            for rev in reviews:
                if not rev.strip():
                    continue
                valid_reviews += 1
                total_len += len(rev.strip())
                total_sentiment += self.analyzer.polarity_scores(rev.strip())['compound']
                
        features['review_count'] = valid_reviews
        features['avg_review_length'] = (total_len / valid_reviews) if valid_reviews > 0 else 0
        features['avg_sentiment_score'] = (total_sentiment / valid_reviews) if valid_reviews > 0 else 0.0
        
        df_input = pd.DataFrame([features])
        prediction = self.opt_model.predict(df_input)[0]
        
        return prediction, {
            'review_count': valid_reviews,
            'avg_sentiment': features['avg_sentiment_score']
        }

    def generate_recommendations(self, current_rating, pasted_reviews_text="", **params):
        """
        Runs counterfactual simulations to find the highest impact business change.
        """
        recommendations = []
        
        # Test 1: Add Table Booking
        if not params.get('book_table', False):
            sim_params = params.copy()
            sim_params['book_table'] = True
            sim_rating, _ = self.optimize_existing_restaurant(pasted_reviews_text, **sim_params)
            diff = sim_rating - current_rating
            if diff >= 0.10:
                recommendations.append({'text': 'Add Table Booking to your restaurant.', 'impact': diff})
                
        # Test 2: Add Online Ordering
        if not params.get('online_order', False):
            sim_params = params.copy()
            sim_params['online_order'] = True
            sim_rating, _ = self.optimize_existing_restaurant(pasted_reviews_text, **sim_params)
            diff = sim_rating - current_rating
            if diff >= 0.10:
                recommendations.append({'text': 'Enable Online Ordering for delivery.', 'impact': diff})
                
        # Test 3: List Signature Dishes
        if int(params.get('num_liked_dishes', 0)) == 0:
            sim_params = params.copy()
            sim_params['num_liked_dishes'] = 3
            sim_rating, _ = self.optimize_existing_restaurant(pasted_reviews_text, **sim_params)
            diff = sim_rating - current_rating
            if diff >= 0.10:
                recommendations.append({'text': 'List signature "must-try" dishes on your Zomato profile.', 'impact': diff})
                
        # Test 4: Improve Value for Money (Lower cost by 10%)
        current_cost = float(params.get('cost_for_two', 500))
        sim_params = params.copy()
        sim_params['cost_for_two'] = current_cost * 0.90
        sim_rating, _ = self.optimize_existing_restaurant(pasted_reviews_text, **sim_params)
        diff = sim_rating - current_rating
        if diff >= 0.10:
            recommendations.append({'text': f'Consider a 10% price reduction (approx. ₹{int(current_cost * 0.10)}) to improve your Value-For-Money score.', 'impact': diff})
            
        if not recommendations:
            return "Your current configuration is already near-optimal based on the model."
            
        # Sort by highest impact and return the top 1
        recommendations.sort(key=lambda x: x['impact'], reverse=True)
        top_rec = recommendations[0]
        
        return f"Recommendation: {top_rec['text']} (Predicted Impact: +{top_rec['impact']:.2f} stars)"
