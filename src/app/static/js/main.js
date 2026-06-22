let currentMode = 'A';

function switchMode(mode) {
    currentMode = mode;
    
    document.getElementById('btn-mode-a').classList.toggle('active', mode === 'A');
    document.getElementById('btn-mode-b').classList.toggle('active', mode === 'B');
    
    const modeBFields = document.getElementById('mode-b-fields');
    if (mode === 'B') {
        modeBFields.classList.remove('hidden');
    } else {
        modeBFields.classList.add('hidden');
    }
    
    document.getElementById('results-dashboard').classList.add('hidden');
}

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submit-btn');
    const originalText = submitBtn.innerText;
    submitBtn.innerText = 'Calculating Predictions...';
    submitBtn.disabled = true;

    try {
        const rawCuisines = document.getElementById('cuisines').value;
        if (/\d/.test(rawCuisines)) {
            alert('Cuisines cannot contain numbers. Please enter text only.');
            submitBtn.innerText = originalText;
            submitBtn.disabled = false;
            return;
        }
        
        const cuisinesList = rawCuisines
            .split(',')
            .map(c => c.trim())
            .filter(c => c.length > 0);

        const payload = {
            mode: currentMode,
            location: document.getElementById('location').value,
            'listed_in(city)': 'Koramangala',
            rest_type: document.getElementById('rest_type').value,
            'listed_in(type)': 'Dine-out',
            cost_for_two: parseFloat(document.getElementById('cost_for_two').value),
            online_order: document.getElementById('online_order').checked,
            book_table: document.getElementById('book_table').checked,
            cuisines: cuisinesList,
            is_chain: false,
            chain_outlets: 0
        };

        if (currentMode === 'B') {
            payload.votes = parseInt(document.getElementById('votes').value) || 0;
            payload.num_liked_dishes = parseInt(document.getElementById('num_liked_dishes').value) || 0;
            payload.pasted_reviews = document.getElementById('pasted_reviews').value;
        }

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            document.getElementById('results-dashboard').classList.remove('hidden');
            
            const ratingDisplay = document.getElementById('rating-display');
            ratingDisplay.innerText = data.rating.toFixed(2) + ' / 5.0';
            
            if (data.rating >= 4.0) ratingDisplay.style.color = '#10b981'; 
            else if (data.rating >= 3.0) ratingDisplay.style.color = '#fbbf24'; 
            else ratingDisplay.style.color = '#ef4444'; 

            const nlpStats = document.getElementById('nlp-stats-container');
            const recBox = document.getElementById('recommendation-box');
            
            if (currentMode === 'B' && data['nlp-stats']) {
                nlpStats.classList.remove('hidden');
                document.getElementById('review-count').innerText = data['nlp-stats'].review_count;
                document.getElementById('sentiment-score').innerText = data['nlp-stats'].avg_sentiment.toFixed(2);
                
                if (data.recommendation) {
                    recBox.classList.remove('hidden');
                    document.getElementById('recommendation-text').innerText = data.recommendation;
                } else {
                    recBox.classList.add('hidden');
                }
            } else {
                nlpStats.classList.add('hidden');
                recBox.classList.add('hidden');
            }
        } else {
            alert('Prediction failed: ' + data.error);
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the Prediction Server.');
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
    }
});
