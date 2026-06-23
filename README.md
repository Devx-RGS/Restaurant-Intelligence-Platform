# Restaurant Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-Advanced%20GBDT-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-Backend%20API-lightgrey.svg)
![NLP](https://img.shields.io/badge/NLP-VADER%20Sentiment-green.svg)

An end-to-end Machine Learning Web Application designed to predict restaurant success and generate data-driven business recommendations in the highly competitive Bangalore (Zomato) market.

## Live Demo
[**Click here to view the live project on Hugging Face Spaces!**](https://huggingface.co/spaces/Devx-RGS/Restaurant-Intelligence-Platform)

## Overview
This project takes a raw, noisy dataset of 50,000+ restaurants and transforms it into a production-ready AI tool. It features a **Dual-Engine Architecture** to solve the "Cold Start" problem for new restaurants, while optimizing existing ones using Live NLP Sentiment Analysis.

### Key Features
- **Dual-Engine Predictor:** Uses two separate CatBoost models. One for pre-launch market viability, and one for post-launch active optimization.
- **Live NLP Sentiment:** Integrates VADER NLP to instantly analyze pasted customer reviews and calculate a live sentiment score.
- **Counterfactual Recommendation Engine:** Runs background simulations (e.g., *What if we dropped prices by 10%?*) to generate the highest-impact business advice.
- **Flask API + Glassmorphism UI:** A stateless Python backend serving a highly optimized HTML/CSS/JS frontend.

---

## Project Architecture

### Phase 1: Data Cleaning & Engineering 
- Cleaned 50,000+ noisy rows of Zomato data.
- Handled missing values, formatted cost metrics, and parsed complex comma-separated text strings.
- Applied **Multi-Label One-Hot Encoding** to isolate the Top 15 Cuisines into distinct mathematical features.

### Phase 2: Solving Data Leakage
- Split the training data to isolate **Pre-Launch features** (Location, Cost, Cuisines) from **Post-Launch metrics** (Votes, Reviews).
- Prevented the model from "cheating" by building a strict Cold-Start baseline model.

### Phase 3: Advanced Modeling
- Engineered interaction features (e.g., `cost_x_booktable`).
- Trained an advanced **CatBoost Regressor** capable of handling raw categorical text natively.

### Phase 4: Full-Stack Web App
- Built a **Flask REST API** to handle asynchronous prediction requests.
- Designed a premium, responsive **Glassmorphism UI** using HTML/CSS.
- Implemented real-time Javascript DOM manipulation to display predicted ratings and AI business recommendations.
- Containerized using **Docker** for cloud deployment.

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Restaurant-Intelligence-Platform.git
   cd Restaurant-Intelligence-Platform
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Prediction Server:**
   ```bash
   python src/app/app.py
   ```

4. **Open the Web App:**
   Navigate to `http://127.0.0.1:5000` in your web browser.

<img width="1230" height="881" alt="Screenshot 2026-06-22 224416" src="https://github.com/user-attachments/assets/85b79c2c-8ff4-411b-bb4d-af872efd3c6e" />


<img width="655" height="921" alt="image" src="https://github.com/user-attachments/assets/68474eab-8168-42f8-bdc9-8b1df3e04e8d" />
