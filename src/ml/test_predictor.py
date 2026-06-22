from predictor import ZomatoPredictor

def run_tests():
    predictor = ZomatoPredictor()
    
    # Base parameters for the restaurant
    params = {
        'location': 'Koramangala 5th Block',
        'listed_in(city)': 'Koramangala',
        'rest_type': 'Casual Dining',
        'listed_in(type)': 'Dine-out',
        'cost_for_two': 1000,
        'online_order': True,
        'book_table': False,
        'cuisines': ['Chinese', 'North Indian'],
        'is_chain': False,
        'chain_outlets': 0
    }
    
    # Mode A: New Restaurant Baseline
    print("\n--- MODE A: NEW RESTAURANT ---")
    baseline_rating = predictor.simulate_new_restaurant(**params)
    print(f"Predicted Baseline Rating: {baseline_rating:.2f} / 5.0")
    
    print("\n--- MODE B: EXISTING RESTAURANT ---")
    params.update({'votes': 150, 'num_liked_dishes': 0, 'book_table': False})
    
    fake_reviews = """
    The food was absolutely fantastic! Loved the Chinese.
    Service was a bit slow, but overall a decent place.
    Terrible experience. The North Indian curry was cold.
    Best place in Koramangala! Will definitely come back.
    It was okay, nothing special but not bad either.
    """
    
    opt_rating, nlp_stats = predictor.optimize_existing_restaurant(
        pasted_reviews_text=fake_reviews, 
        **params
    )
    
    print(f"NLP Engine detected {nlp_stats['review_count']} reviews.")
    print(f"Live Sentiment Score: {nlp_stats['avg_sentiment']:.2f}")
    print(f"Predicted Optimized Rating: {opt_rating:.2f} / 5.0")
    
    print("\n--- ACTIONABLE RECOMMENDATION ---")
    recommendation = predictor.generate_recommendations(
        current_rating=opt_rating,
        pasted_reviews_text=fake_reviews,
        **params
    )
    print(recommendation)
    
if __name__ == "__main__":
    run_tests()
