# FootbrAIn - AI Football Prediction & Value Betting Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-70.44%25-brightgreen.svg)

**Advanced AI-powered football prediction engine with value betting strategies**

[Features](#-key-features) • [Installation](#-quick-installation) • [API Docs](#-api-endpoints) • [Strategy](#-betting-strategies)

</div>

---

## 📊 Executive Summary

VARIO is an advanced AI-powered football prediction engine that combines machine learning with value betting strategies. It achieves **70.44% prediction accuracy** using ensemble methods (XGBoost + RandomForest) and provides real-time Expected Value (EV) calculations for profitable betting decisions.

### Key Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **70.44%** |
| Home Win Precision | **81%** |
| Away Win Precision | **72%** |
| Draw Precision | 44% |
| Training Samples | 86,983 matches |
| Features Used | 15+ advanced metrics |
| Database Size | ~700MB |
| API Response Time | <500ms |

---

## 🎯 Key Features

### 🤖 Machine Learning
- **Ensemble Model**: XGBoost + RandomForest stacking classifier
- **70.44% Accuracy**: Validated on 86,983 historical matches
- **15+ Features**: Form, clean sheets, goal averages, team value, ELO ratings
- **Auto-Retraining**: Weekly pipeline with cron jobs
- **Multi-Version Support**: v4 (stable) and v5 (enhanced) with auto-fallback

### 💰 Value Betting System
- **EV Calculation**: Automated Expected Value for each outcome
- **Kelly Criterion**: Optimal stake sizing with fractional Kelly
- **3 Strategies**: Conservative, Balanced, Aggressive
- **Probability Thresholds**: Minimum 45% for home, 35% for away bets
- **Draw Penalty**: +10% EV required for draw bets (model weakness)

### 🎨 Web Interface
- **Team Search**: Type "oviedo" finds "Real Oviedo S.A.D"
- **Real-time Predictions**: Instant probability updates
- **Visual Analytics**: Form circles, probability bars, feature importance
- **Referee Stats**: Historical card averages and style classification
- **Mobile Responsive**: Works on all devices

### 📈 Training Reports
- Confusion Matrix
- Feature Importance Plot
- Permutation Importance
- SHAP Summary (optional)
- Cross-validation scores

---

## 🚀 Quick Installation

### Prerequisites

```bash
# System Requirements
- Ubuntu 20.04+ / Debian 11+ / Linux Ubuntu Server
- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for data download
```
# Step 1: Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev sqlite3 curl unzip
sudo apt-get install -y build-essential cmake  # For LightGBM (optional)
```

# Step 2: Clone and Setup Project
```bash
sudo mkdir -p /var/www/footbrain
cd /var/www/footbrain
git clone https://github.com/yourusername/footbrAIn.git
```

# Step 3: Install Python Packages
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install core packages
pip install --upgrade pip
pip install fastapi uvicorn pandas numpy scikit-learn
pip install xgboost joblib matplotlib seaborn
pip install imbalanced-learn sqlalchemy python-multipart

# Optional but recommended
pip install lightgbm catboost shap
```

# Step 4: Download and Import Data
```bash
# Make run script executable
chmod +x run.sh

# Download and import football data (~700MB)
bash run.sh

# Expected output:
# [1/5] Preparing workspace...
# [2/5] Downloading dataset...
# [3/5] Checking for existing database...
# [4/5] Unzipping files...
# [5/5] Running data import...
# ✅ All operations completed successfully!
```

# Step 5: Train the Model
```bash
# Train the v4 model (stable, 70.44% accuracy)
python3 v4.py

# Expected output:
# ============================================================
# ENHANCED VARIO MODEL TRAINING v5.0
# Target: 75%+ Accuracy with Improved Draw Prediction
# ============================================================
# Step 1: Loading games with enhanced columns...
# Step 2: Loading clubs with detailed stats...
# ...
# ✅ Overall Accuracy: 70.44%
# ✅ Model saved to models/match_outcome_model_v4.pkl
```
# Step 6: Set Up API Service
```bash
sudo nano /etc/systemd/system/footbrain-api.service
```
Add this content:
```bash
[Unit]
Description=VARIO Football Prediction API
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/footbrain
Environment="PATH=/var/www/footbrain/venv/bin"
ExecStart=/var/www/footbrain/venv/bin/uvicorn main:app --host 0.0.0.0 --port 5002
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable footbrain-api
sudo systemctl start footbrain-api
sudo systemctl status footbrain-api

# Expected output:
# ● footbrain-api.service - VARIO Football Prediction API
#    Loaded: loaded (/etc/systemd/system/footbrain-api.service; enabled)
#    Active: active (running)
```

# Step 7: Configure Web Interface
```bash
# Ensure static directory exists
mkdir -p /var/www/footbrain/static

# Copy index.html if not already there
cp static/index.html /var/www/footbrain/static/

# Set permissions
sudo chown -R www-data:www-data /var/www/footbrain
sudo chmod -R 755 /var/www/footbrain
```

# Step 8: Set Up Automated Training (Optional)
```bash
crontab -e

# Add this line (runs every Thursday at 12:00 PM)
0 12 * * 4 /usr/bin/python3 /var/www/footbrain/pipeline.py >> /var/log/footbrain/cron.log 2>&1
```

# Step 9: Verify Installation
```bash
# Test API
curl http://localhost:5002/api/health

# Expected response:
# {"status":"healthy","version":"0.75","model_loaded":true,"cards_model_loaded":true}
```

# 📖 Usage Guide
```bash
http://your-server-ip:5002
```

Features:
🔍 Team Search: Type partial names (e.g., "oviedo" → "Real Oviedo S.A.D")

📊 Real-time Predictions: Probability bars update instantly

💰 Value Betting: Automatic EV calculation with stake recommendations

🃏 Referee Stats: Card history and style classification

📈 Training Reports: Model performance visualizations

🎯 Strategy Selection: Conservative/Balanced/Aggressive

# API Endpoints
```bash
curl http://localhost:5002/api/health
```

# Get Match Prediction
```bash
curl -X POST http://localhost:5002/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "FC Barcelona",
    "away_team": "Real Madrid",
    "referee": "Michael Oliver"
  }'
```
# Response
```bash
{
  "home_team": "FC Barcelona",
  "away_team": "Real Madrid",
  "venue": "Camp Nou",
  "referee": "Michael Oliver",
  "prediction": "Home Win",
  "probabilities": {
    "home_win": 0.58,
    "draw": 0.22,
    "away_win": 0.20
  },
  "confidence_level": "Medium Confidence"
}
```

# Get Betting Analysis
```bash
curl -X POST "http://localhost:5002/api/betting/analyze?bankroll=1000&strategy=balanced" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "FC Barcelona",
    "away_team": "Real Madrid",
    "referee": "Michael Oliver"
  }'
```
# Response
```bash
{
  "betting_analysis": {
    "has_value_bet": true,
    "best_bet": {
      "outcome_display": "🏠 Home Win",
      "model_probability": 0.58,
      "bookmaker_odds": 1.85,
      "expected_value_percent": 7.3,
      "recommended_stake": 18.25,
      "confidence": "MEDIUM"
    }
  }
}
```

# Get All Teams
```bash
curl http://localhost:5002/api/teams
```

# Get Referee Statistics
```bash
curl http://localhost:5002/api/referee/Michael%20Oliver
```
#Response
```bash
{
  "total_games": 127,
  "avg_cards": 4.2,
  "total_cards": 533,
  "recent_games": [...]
}
```

#Get Model Information
```bash
curl http://localhost:5002/api/model/info
```

🎲 Betting Strategies
Strategy Comparison
Strategy	Min EV	Kelly Fraction	Max Stake	Min Home Prob	Min Away Prob	Description
🛡️ Conservative	8%	15%	3%	50%	40%	Low risk, fewer bets, suitable for beginners
⚖️ Balanced	5%	25%	5%	45%	35%	Optimal for 70% accuracy model
⚡ Aggressive	3%	40%	8%	40%	30%	Higher risk, more betting opportunities

# Value Betting Formula
```bash
# Expected Value Calculation
EV = (Model_Probability × Bookmaker_Odds) - 1

# Decision Rule
if EV > min_threshold:
    place_bet()
else:
    skip_bet()
```

# Kelly Criterion Stake Sizing
```bash
# Full Kelly
Kelly_% = (Odds × Probability - 1) / (Odds - 1)

# Fractional Kelly (safer)
Final_Stake = Bankroll × (Kelly_% × Kelly_Fraction)

# Example with $1000 bankroll
# Probability: 58%, Odds: 1.85, Kelly_Fraction: 0.25
Kelly_% = (1.85 × 0.58 - 1) / (1.85 - 1) = 0.073 = 7.3%
Final_Stake = $1000 × (0.073 × 0.25) = $18.25
```
# Draw Betting Rule
Due to model weakness (44% precision), draws require:

Minimum 30% probability

+10% higher EV than home/away bets

Extra caution in strategy selection

📁 Project Structure
```bash
/var/www/footbrain/
├── app/
│   ├── __init__.py
│   ├── database.py              # SQLite operations
│   ├── predictors.py            # ML model predictors
│   ├── betting_analyzer.py      # EV & Kelly calculations
│   └── betting_strategy.py      # Strategy configs
│
├── models/
│   ├── match_outcome_model_v4.pkl       # Main model (70.44%)
│   ├── match_outcome_model_v4_features.pkl
│   ├── events_outcome_model.pkl         # Cards prediction
│   ├── scaler_v5.pkl                    # Feature scaler
│   └── selector_v5.pkl                  # Feature selector
│
├── static/
│   ├── index.html               # Web interface
│   ├── v4_confusion_matrix.png
│   ├── v4_feature_importance.png
│
├── main.py                      # FastAPI application (500+ lines)
├── v4.py                        # Model training script (600+ lines)
├── pipeline.py                  # Automated training pipeline
├── run.sh                       # Data download script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```
🧠 How It Works
Feature Engineering Pipeline
text
Raw Data → Feature Engineering → Model Training → Prediction
1. Team Form (Weighted)
python
weights = [0.5, 0.6, 0.8, 0.9, 1.0]  # Most recent game highest weight
form = (points × weights).sum() / weights.sum()
2. Clean Sheets
python
home_clean_sheets = (away_goals == 0).rolling(5).sum()
away_clean_sheets = (home_goals == 0).rolling(5).sum()
3. Goal Averages
python
home_scored_avg = home_goals.rolling(5).mean()
home_conceded_avg = away_goals.rolling(5).mean()
4. Relative Strength
python
relative_strength = (home_form + 0.1) / (away_form + 0.1)
5. Position Similarity
python
position_similarity = 1 - abs(home_pos - away_pos) / 20
