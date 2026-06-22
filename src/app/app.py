from flask import Flask, request, jsonify, render_template
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml.predictor import ZomatoPredictor

app = Flask(__name__, static_folder = 'static', template_folder = 'templates')

# Initialize the CatBoost and NLP Prediction Engine on server startup
print("Booting up the Predictor Engine")
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
predictor = ZomatoPredictor(models_dir = models_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods = ['GET'])
def health_check():
    """Simple health check endpoint for monitoring deployment status."""
    return jsonify({
        'status': 'ok',
        'message': "Predictor Engine is running"
    })

@app.route('/api/predict', methods = ['POST'])
def predict():
    """
    Main prediction endpoint.
    Expects a JSON payload containing restaurant features.
    Routes to either the Cold Start model (Mode A) or Optimizer model (Mode B).
    """
    try:
        data = request.json
        mode = data.get('mode', 'A')

        if mode == "A":
            # Mode A: Pre-launch predictions
            rating = predictor.simulate_new_restaurant(**data)
            return jsonify({
                'success': True,
                'rating': round(rating, 2),
                'nlp_stats' : None,
                'recommendation': None
            })

        elif mode == 'B':
            # Mode B: Post-launch optimization with live NLP and recommendations
            pasted_reviews = data.get('pasted_reviews', '')
            rating, nlp_stats = predictor.optimize_existing_restaurant(pasted_reviews, **data)

            # Generate counterfactual business recommendations
            recommendation = predictor.generate_recommendations(rating, pasted_reviews, **data)

            return jsonify({
                'success': True,
                'rating': round(rating, 2),
                'nlp-stats': nlp_stats,
                'recommendation': recommendation
            })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug = True, port = 5000)