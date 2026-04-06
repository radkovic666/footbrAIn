from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
import sys
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.database import (
    get_club_info, get_all_teams, get_all_referees, get_all_venues,
    get_teams_by_country, get_all_countries, get_referee_stats
)
from app.predictors import MatchPredictor, CardsPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VARIO Football Prediction Engine",
    description="Advanced football match outcome prediction system with Value Betting",
    version="0.75",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

match_predictor = None
cards_predictor = None
feature_names = []


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    referee: Optional[str] = None
    venue: Optional[str] = None


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    venue: str
    referee: Optional[str]
    prediction: str
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    cards_prediction: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


@app.on_event("startup")
async def startup_event():
    global match_predictor, cards_predictor, feature_names
    
    logger.info("Loading models...")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    model_path = os.path.join(models_dir, "match_outcome_model_v4.pkl")
    features_path = os.path.join(models_dir, "match_outcome_model_v4_features.pkl")
    cards_model_path = os.path.join(models_dir, "events_outcome_model.pkl")
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        try:
            match_predictor = MatchPredictor(model_path, features_path)
            feature_names = match_predictor.get_feature_names()
            logger.info(f"Match model loaded with {len(feature_names)} features")
        except Exception as e:
            logger.error(f"Error loading match model: {e}")
    else:
        logger.warning(f"Match model not found at {model_path}")
    
    if os.path.exists(cards_model_path):
        try:
            cards_predictor = CardsPredictor(cards_model_path)
            logger.info("Cards model loaded")
        except Exception as e:
            logger.error(f"Error loading cards model: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding='utf-8') as f:
            content = f.read()
            return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>VARIO Football Prediction Engine</h1><p>Loading interface...</p>")


@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon to stop loading spinner"""
    from fastapi.responses import Response
    return Response(content="", media_type="image/x-icon")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": "0.75",
        "model_loaded": match_predictor is not None,
        "cards_model_loaded": cards_predictor is not None,
        "feature_count": len(feature_names) if feature_names else 0
    }


@app.get("/api/teams")
async def get_teams():
    try:
        teams = get_all_teams()
        return {"teams": teams, "count": len(teams)}
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/teams/by-country")
async def get_teams_grouped():
    try:
        teams_by_country = get_teams_by_country()
        countries = list(teams_by_country.keys())
        return {"countries": countries, "teams_by_country": teams_by_country}
    except Exception as e:
        logger.error(f"Error getting grouped teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/countries")
async def get_countries():
    try:
        countries = get_all_countries()
        return {"countries": countries}
    except Exception as e:
        logger.error(f"Error getting countries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/referees")
async def get_referees():
    try:
        referees = get_all_referees()
        return {"referees": referees, "count": len(referees)}
    except Exception as e:
        logger.error(f"Error getting referees: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/referee/{referee_name}")
async def get_referee_details(referee_name: str):
    """Get detailed statistics for a specific referee"""
    try:
        stats = get_referee_stats(referee_name)
        return stats
    except Exception as e:
        logger.error(f"Error getting referee stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/venues")
async def get_venues():
    try:
        venues = get_all_venues()
        return {"venues": venues, "count": len(venues)}
    except Exception as e:
        logger.error(f"Error getting venues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if match_predictor is None:
        raise HTTPException(status_code=503, detail="Match model not loaded")
    
    try:
        home_club = get_club_info(request.home_team)
        away_club = get_club_info(request.away_team)
        
        if not home_club:
            raise HTTPException(status_code=404, detail=f"Home team '{request.home_team}' not found")
        if not away_club:
            raise HTTPException(status_code=404, detail=f"Away team '{request.away_team}' not found")
        
        result = match_predictor.predict(
            home_club=home_club,
            away_club=away_club,
            referee=request.referee,
            venue=request.venue
        )
        
        if cards_predictor and request.referee:
            cards_result = cards_predictor.predict(
                home_club=home_club,
                away_club=away_club,
                referee=request.referee
            )
            result["cards_prediction"] = cards_result
        
        return PredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            venue=request.venue or home_club.get("stadium_name", "Unknown"),
            referee=request.referee,
            prediction=result["outcome"],
            probabilities=result["probabilities"],
            features_used=result["features"],
            cards_prediction=result.get("cards_prediction")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/betting/analyze")
async def betting_analysis(request: PredictionRequest, 
                          bankroll: float = 1000,
                          strategy: str = "balanced"):
    """Get betting recommendation with value analysis"""
    if match_predictor is None:
        raise HTTPException(status_code=503, detail="Match model not loaded")
    
    try:
        home_club = get_club_info(request.home_team)
        away_club = get_club_info(request.away_team)
        
        if not home_club:
            raise HTTPException(status_code=404, detail=f"Home team '{request.home_team}' not found")
        if not away_club:
            raise HTTPException(status_code=404, detail=f"Away team '{request.away_team}' not found")
        
        # Get regular prediction first
        result = match_predictor.predict(
            home_club=home_club,
            away_club=away_club,
            referee=request.referee,
            venue=request.venue
        )
        
        # Get cards prediction if available
        cards_result = None
        if cards_predictor and request.referee:
            cards_result = cards_predictor.predict(
                home_club=home_club,
                away_club=away_club,
                referee=request.referee
            )
            result["cards_prediction"] = cards_result
        
        # Add betting analysis
        probs = result['probabilities']
        
        # Estimate market odds (with margin)
        estimated_odds = {
            'home_win': round(1 / (probs['home_win'] * 0.94), 2),
            'draw': round(1 / (probs['draw'] * 0.94), 2),
            'away_win': round(1 / (probs['away_win'] * 0.94), 2)
        }
        
        # Calculate EV for each outcome
        ev_home = (probs['home_win'] * estimated_odds['home_win']) - 1
        ev_draw = (probs['draw'] * estimated_odds['draw']) - 1
        ev_away = (probs['away_win'] * estimated_odds['away_win']) - 1
        
        # CRITICAL FIX: Minimum probability thresholds
        # Based on your model's performance (70% accuracy)
        MIN_PROB_HOME = 0.45  # Home wins below 45% are too risky
        MIN_PROB_DRAW = 0.30  # Draws below 30% - model is weak here
        MIN_PROB_AWAY = 0.35  # Away wins below 35% - don't bet
        
        # Apply strategy adjustments
        if strategy == "conservative":
            min_ev = 0.08
            kelly_fraction = 0.15
            max_stake_pct = 0.03
            draw_penalty = 0.15
            MIN_PROB_HOME = 0.50
            MIN_PROB_AWAY = 0.40
        elif strategy == "aggressive":
            min_ev = 0.03
            kelly_fraction = 0.40
            max_stake_pct = 0.08
            draw_penalty = 0.05
            MIN_PROB_HOME = 0.40
            MIN_PROB_AWAY = 0.30
        else:  # balanced
            min_ev = 0.05
            kelly_fraction = 0.25
            max_stake_pct = 0.05
            draw_penalty = 0.10
            MIN_PROB_HOME = 0.45
            MIN_PROB_AWAY = 0.35
        
        outcomes = []
        
        # Home Win analysis
        if probs['home_win'] >= MIN_PROB_HOME:
            outcomes.append({
                'outcome': 'home_win', 
                'display': '🏠 Home Win', 
                'ev': ev_home, 
                'prob': probs['home_win'], 
                'odds': estimated_odds['home_win'],
                'penalty': 0,
                'meets_prob_threshold': True
            })
        else:
            outcomes.append({
                'outcome': 'home_win', 
                'display': '🏠 Home Win', 
                'ev': ev_home, 
                'prob': probs['home_win'], 
                'odds': estimated_odds['home_win'],
                'penalty': 0,
                'meets_prob_threshold': False,
                'skip_reason': f'Probability too low ({probs["home_win"]*100:.1f}% < {MIN_PROB_HOME*100:.0f}%)'
            })
        
        # Draw analysis (extra penalty because model is weak here)
        draw_min_prob = 0.30
        if probs['draw'] >= draw_min_prob:
            outcomes.append({
                'outcome': 'draw', 
                'display': '🤝 Draw', 
                'ev': ev_draw - draw_penalty, 
                'raw_ev': ev_draw,
                'prob': probs['draw'], 
                'odds': estimated_odds['draw'],
                'penalty': draw_penalty,
                'meets_prob_threshold': True
            })
        else:
            outcomes.append({
                'outcome': 'draw', 
                'display': '🤝 Draw', 
                'ev': ev_draw - draw_penalty, 
                'raw_ev': ev_draw,
                'prob': probs['draw'], 
                'odds': estimated_odds['draw'],
                'penalty': draw_penalty,
                'meets_prob_threshold': False,
                'skip_reason': f'Draw probability too low ({probs["draw"]*100:.1f}% < 30%) - Model is weak on draws'
            })
        
        # Away Win analysis
        if probs['away_win'] >= MIN_PROB_AWAY:
            outcomes.append({
                'outcome': 'away_win', 
                'display': '🚗 Away Win', 
                'ev': ev_away, 
                'prob': probs['away_win'], 
                'odds': estimated_odds['away_win'],
                'penalty': 0,
                'meets_prob_threshold': True
            })
        else:
            outcomes.append({
                'outcome': 'away_win', 
                'display': '🚗 Away Win', 
                'ev': ev_away, 
                'prob': probs['away_win'], 
                'odds': estimated_odds['away_win'],
                'penalty': 0,
                'meets_prob_threshold': False,
                'skip_reason': f'Probability too low ({probs["away_win"]*100:.1f}% < {MIN_PROB_AWAY*100:.0f}%) - Long shots are dangerous'
            })
        
        # Filter for value bets that ALSO meet probability threshold
        valid_bets = []
        for o in outcomes:
            effective_ev = o['ev'] - o.get('penalty', 0)
            if o['meets_prob_threshold'] and effective_ev > min_ev:
                valid_bets.append({
                    **o,
                    'effective_ev': effective_ev
                })
        
        valid_bets.sort(key=lambda x: x['effective_ev'], reverse=True)
        best = valid_bets[0] if valid_bets else None
        has_value = best is not None
        
        # Kelly calculation only for valid bets
        recommended_stake = 0
        if has_value and best['odds'] > 1:
            kelly = (best['odds'] * best['prob'] - 1) / (best['odds'] - 1)
            safe_kelly = max(0, min(kelly * kelly_fraction, max_stake_pct))
            recommended_stake = round(bankroll * safe_kelly, 2)
        
        # Get confidence level (based on probability AND EV)
        if has_value:
            if best['prob'] > 0.55 and best['effective_ev'] > 0.10:
                confidence = "HIGH"
            elif best['prob'] > 0.45 and best['effective_ev'] > 0.07:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            confidence = "NONE"
        
        # Build skip reasons for display
        skip_reasons = []
        for o in outcomes:
            if not o.get('meets_prob_threshold', True):
                skip_reasons.append(o.get('skip_reason', f'{o["display"]}: {o["prob"]*100:.1f}% probability'))
        
        strategy_names = {
            "conservative": "🛡️ Conservative",
            "balanced": "⚖️ Balanced",
            "aggressive": "⚡ Aggressive"
        }
        
        betting_analysis = {
            'has_value_bet': has_value,
            'best_bet': {
                'outcome': best['outcome'],
                'outcome_display': best['display'],
                'model_probability': round(best['prob'], 3),
                'bookmaker_odds': best['odds'],
                'expected_value': round(best['ev'], 3),
                'expected_value_percent': round(best['ev'] * 100, 1),
                'effective_ev_percent': round(best['effective_ev'] * 100, 1),
                'recommended_stake': recommended_stake,
                'stake_percent': round((recommended_stake / bankroll) * 100, 2) if bankroll > 0 else 0,
                'confidence': confidence
            } if has_value else None,
            'all_bets': [
                {
                    'outcome': o['outcome'], 
                    'outcome_display': o['display'], 
                    'probability': round(o['prob'] * 100, 1),
                    'odds': o['odds'],
                    'expected_value': round(o['ev'] * 100, 1),
                    'is_value_bet': o.get('meets_prob_threshold', False) and (o['ev'] - o.get('penalty', 0)) > min_ev,
                    'meets_prob_threshold': o.get('meets_prob_threshold', False),
                    'skip_reason': o.get('skip_reason', None)
                }
                for o in outcomes
            ],
            'skip_reasons': skip_reasons,
            'min_probability_thresholds': {
                'home_win': round(MIN_PROB_HOME * 100, 0),
                'draw': 30,
                'away_win': round(MIN_PROB_AWAY * 100, 0)
            },
            'summary': f"{'✅ VALUE BET FOUND!' if has_value else '❌ No value bet'} - {best['display'] if has_value else 'No qualifying bets'} (EV: +{best['ev']*100:.1f}%)" if has_value else f"❌ NO VALUE BET - {skip_reasons[0] if skip_reasons else 'No bets meet criteria'}"
        }
        
        return {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "venue": result.get("venue", "Unknown"),
            "referee": request.referee,
            "prediction": result["outcome"],
            "probabilities": result["probabilities"],
            "features_used": result.get("features", {}),
            "feature_explanations": result.get("feature_explanations", []),
            "home_form_last5": result.get("home_form_last5", []),
            "away_form_last5": result.get("away_form_last5", []),
            "home_form_description": result.get("home_form_description", ""),
            "away_form_description": result.get("away_form_description", ""),
            "confidence_level": result.get("confidence_level", ""),
            "betting_analysis": betting_analysis,
            "strategy_used": strategy_names.get(strategy, "⚖️ Balanced"),
            "cards_prediction": cards_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Betting analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports")
async def get_reports():
    reports_dir = static_dir
    reports = []
    
    if os.path.exists(reports_dir):
        for f in os.listdir(reports_dir):
            if f.endswith('.png'):
                reports.append({
                    "name": f.replace('.png', '').replace('_', ' ').title(),
                    "url": f"/static/{f}",
                    "modified": datetime.fromtimestamp(os.path.getmtime(os.path.join(reports_dir, f))).isoformat()
                })
    
    return {"reports": reports}


@app.get("/api/model/info")
async def model_info():
    return {
        "model_type": "StackingClassifier (XGBoost + RandomForest)",
        "features": feature_names,
        "feature_count": len(feature_names),
        "classes": ["Home Win", "Draw", "Away Win"],
        "current_season": "2025 (from August 1, 2025)",
        "accuracy": "70.44%",
        "training_samples": 86983
    }