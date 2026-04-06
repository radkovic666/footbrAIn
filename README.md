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
# Create project directory
```bash
sudo mkdir -p /var/www/footbrain
cd /var/www/footbrain
```

# Clone repository (or copy files manually)
```bash
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
