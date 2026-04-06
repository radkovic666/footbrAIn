import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.database import get_latest_position, get_games_history, get_last_5_games_with_details


class MatchPredictor:
    """Match outcome predictor - uses current season data only for predictions"""

    def get_betting_recommendation(self,
                                  home_club: Dict,
                                  away_club: Dict,
                                  bookmaker_odds: Optional[Dict] = None,
                                  bankroll: float = 1000) -> Dict:
        """Get betting recommendation based on model predictions"""

        # Get model predictions
        prediction = self.predict(home_club, away_club)

        # If no bookmaker odds provided, use simulated/typical odds
        if not bookmaker_odds:
            bookmaker_odds = self._estimate_market_odds(
                prediction['probabilities'],
                prediction.get('features', {})
            )

        # Analyze betting value
        from app.betting_analyzer import BettingAnalyzer
        analyzer = BettingAnalyzer()

        betting_analysis = analyzer.analyze_bet(
            model_probs={
                'home_win': prediction['probabilities']['home_win'],
                'draw': prediction['probabilities']['draw'],
                'away_win': prediction['probabilities']['away_win']
            },
            bookmaker_odds=bookmaker_odds,
            bankroll=bankroll
        )

        prediction['betting_analysis'] = betting_analysis
        return prediction

    def _estimate_market_odds(self, probs: Dict, features: Dict) -> Dict:
        """Estimate market odds based on model probabilities and market inefficiencies"""

        market_bias = 0.02

        market_probs = {
            'home_win': max(0.1, probs['home_win'] + market_bias),
            'draw': max(0.1, probs['draw'] - market_bias / 2),
            'away_win': max(0.1, probs['away_win'] - market_bias / 2)
        }

        total = sum(market_probs.values())
        market_probs = {k: v / total for k, v in market_probs.items()}

        margin = 1.06
        odds = {k: round(margin / v, 2) for k, v in market_probs.items()}

        return odds

    def __init__(self, model_path: str, features_path: str):
        self.model = joblib.load(model_path)
        feature_info = joblib.load(features_path)
        self.feature_names = feature_info['feature_names']
        self.current_season = "2025"
        self.current_season_start = datetime(2025, 8, 1)

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_feature_explanation(self, feature_name: str) -> Dict[str, str]:
        explanations = {
            'position_diff': {'label': 'League Position Difference', 'icon': '📊',
                              'description': 'Higher value means home team is better positioned in league'},
            'form_diff': {'label': 'Recent Form Difference', 'icon': '⚡',
                          'description': 'Positive means home team has better recent results'},
            'clean_sheets_diff': {'label': 'Defensive Strength', 'icon': '🛡️',
                                  'description': 'Number of clean sheets in last 5 games'},
            'home_club_position': {'label': 'Home Team Position', 'icon': '🏠',
                                   'description': 'Home team\'s current league ranking'},
            'away_club_position': {'label': 'Away Team Position', 'icon': '🚗',
                                   'description': 'Away team\'s current league ranking'},
            'form_x_position_home': {'label': 'Home Momentum Score', 'icon': '📈',
                                     'description': 'Combined form and position advantage'},
            'form_x_position_away': {'label': 'Away Momentum Score', 'icon': '📉',
                                     'description': 'Combined form and position for away team'},
            'home_gk_clean_sheets_last5': {'label': 'Home Goalkeeper Form', 'icon': '🧤',
                                           'description': 'Clean sheets by home keeper in last 5'},
            'away_gk_clean_sheets_last5': {'label': 'Away Goalkeeper Form', 'icon': '🧤',
                                           'description': 'Clean sheets by away keeper in last 5'},
            'relative_strength': {'label': 'Relative Strength', 'icon': '💪',
                                  'description': 'Higher than 1 means home team stronger'},
            'win_rate_diff': {'label': 'Win Rate Difference', 'icon': '🏆',
                              'description': 'Recent winning percentage advantage'},
            'away_conceded': {'label': 'Away Defense Rating', 'icon': '🔒',
                              'description': 'Average goals conceded by away team'},
            'home_conceded': {'label': 'Home Defense Rating', 'icon': '🔒',
                              'description': 'Average goals conceded by home team'},
            'trend_diff': {'label': 'Form Trend', 'icon': '📊',
                           'description': 'Whether team is improving or declining'},
            'power_rank_diff': {'label': 'Power Ranking', 'icon': '⚡',
                                'description': 'Based on average goals scored'}
        }

        return explanations.get(feature_name, {
            'label': feature_name,
            'icon': '📊',
            'description': ''
        })
    
    def get_last_5_matches_with_form(self, club_id: int) -> List[Dict]:
        """Get last 5 matches with form indicators (W/D/L)"""
        games = get_last_5_games_with_details(club_id, self.current_season)
        
        if len(games) < 3:
            games = get_last_5_games_with_details(club_id, "2024")
        
        result_map = {0: 'L', 1: 'D', 2: 'W'}
        color_map = {'W': 'green', 'D': 'orange', 'L': 'red'}
        
        for game in games:
            if game['goals_for'] > game['goals_against']:
                game['result'] = 'W'
            elif game['goals_for'] == game['goals_against']:
                game['result'] = 'D'
            else:
                game['result'] = 'L'
            game['result_color'] = color_map[game['result']]
            
        return games[:5]
    
    def calculate_form_last5(self, club_id: int) -> float:
        """Calculate points per game for last 5 matches - current season only"""
        games = get_games_history(club_id, limit=10, season=self.current_season)
        
        if len(games) < 3:
            games = get_games_history(club_id, limit=10, season="2024")
        
        if len(games) < 5:
            return 1.0
        
        points = 0
        count = 0
        
        for game in games[:5]:
            if game['goals_for'] > game['goals_against']:
                points += 3
            elif game['goals_for'] == game['goals_against']:
                points += 1
            count += 1
        
        return points / count if count > 0 else 1.0
    
    def calculate_clean_sheets_last5(self, club_id: int) -> int:
        """Calculate clean sheets in last 5 games"""
        games = get_games_history(club_id, limit=10, season=self.current_season)
        
        if len(games) < 3:
            games = get_games_history(club_id, limit=10, season="2024")
        
        clean_sheets = 0
        for game in games[:5]:
            if game['goals_against'] == 0:
                clean_sheets += 1
        
        return clean_sheets
    
    def calculate_win_rate_last5(self, club_id: int) -> float:
        """Calculate win rate in last 5 games"""
        games = get_games_history(club_id, limit=10, season=self.current_season)
        
        if len(games) < 3:
            games = get_games_history(club_id, limit=10, season="2024")
        
        if len(games) < 5:
            return 0.2
        
        wins = sum(1 for game in games[:5] if game['goals_for'] > game['goals_against'])
        return wins / 5
    
    def calculate_goals_avg_last5(self, club_id: int, is_home_goals: bool = True) -> float:
        """Calculate average goals scored or conceded in last 5 games"""
        games = get_games_history(club_id, limit=10, season=self.current_season)
        
        if len(games) < 3:
            games = get_games_history(club_id, limit=10, season="2024")
        
        if len(games) < 5:
            return 1.0
        
        if is_home_goals:
            goals_sum = sum(game['goals_for'] for game in games[:5])
        else:
            goals_sum = sum(game['goals_against'] for game in games[:5])
        
        return goals_sum / 5
    
    def get_form_strength_description(self, value: float) -> str:
        """Convert numeric form to description"""
        if value >= 2.5:
            return "Excellent 🔥"
        elif value >= 2.0:
            return "Good ✅"
        elif value >= 1.5:
            return "Decent 📈"
        elif value >= 1.0:
            return "Average 📊"
        else:
            return "Poor ⚠️"
    
    def predict(self, home_club: Dict, away_club: Dict, 
                referee: Optional[str] = None, 
                venue: Optional[str] = None) -> Dict[str, Any]:
        """Make match prediction using current season data"""
        
        home_id = home_club['club_id']
        away_id = away_club['club_id']
        
        # Calculate features using current season data only
        home_form = self.calculate_form_last5(home_id)
        away_form = self.calculate_form_last5(away_id)
        form_diff = home_form - away_form
        
        home_clean = self.calculate_clean_sheets_last5(home_id)
        away_clean = self.calculate_clean_sheets_last5(away_id)
        clean_sheets_diff = home_clean - away_clean
        
        home_position = get_latest_position(home_id, self.current_season) or 10.0
        away_position = get_latest_position(away_id, self.current_season) or 10.0
        
        # Interaction terms
        form_x_position_home = home_form * (1.0 / max(home_position, 1))
        form_x_position_away = away_form * (1.0 / max(away_position, 1))
        
        # Relative strength
        relative_strength = (home_form + 0.1) / (away_form + 0.1)
        
        # Win rate difference
        home_win_rate = self.calculate_win_rate_last5(home_id)
        away_win_rate = self.calculate_win_rate_last5(away_id)
        win_rate_diff = home_win_rate - away_win_rate
        
        # Goals conceded
        home_conceded = self.calculate_goals_avg_last5(home_id, is_home_goals=False)
        away_conceded = self.calculate_goals_avg_last5(away_id, is_home_goals=False)
        
        # Goals scored (power ranking)
        home_power = self.calculate_goals_avg_last5(home_id, is_home_goals=True)
        away_power = self.calculate_goals_avg_last5(away_id, is_home_goals=True)
        power_rank_diff = home_power - away_power
        
        # Position difference
        position_diff = home_position - away_position
        
        # Form trend
        home_trend = self.calculate_form_trend(home_id)
        away_trend = self.calculate_form_trend(away_id)
        trend_diff = home_trend - away_trend
        
        # Create feature vector
        features = {
            'position_diff': position_diff,
            'form_diff': form_diff,
            'clean_sheets_diff': clean_sheets_diff,
            'home_club_position': home_position,
            'away_club_position': away_position,
            'form_x_position_home': form_x_position_home,
            'form_x_position_away': form_x_position_away,
            'home_gk_clean_sheets_last5': home_clean,
            'away_gk_clean_sheets_last5': away_clean,
            'relative_strength': relative_strength,
            'power_rank_diff': power_rank_diff,
            'win_rate_diff': win_rate_diff,
            'away_conceded': away_conceded,
            'home_conceded': home_conceded,
            'trend_diff': trend_diff
        }
        
        # Get last 5 matches for display
        home_last5 = self.get_last_5_matches_with_form(home_id)
        away_last5 = self.get_last_5_matches_with_form(away_id)
        
        # Create user-friendly feature explanations
        feature_explanations = []
        for feat_name in self.feature_names:
            if feat_name in features:
                explanation = self.get_feature_explanation(feat_name)
                value = features[feat_name]
                
                # Add context based on value
                if feat_name == 'relative_strength':
                    context = "Home advantage" if value > 1.1 else "Away advantage" if value < 0.9 else "Balanced"
                elif feat_name == 'form_diff':
                    context = "Home in better form" if value > 0.3 else "Away in better form" if value < -0.3 else "Similar form"
                elif feat_name == 'clean_sheets_diff':
                    context = "Home defense stronger" if value > 1 else "Away defense stronger" if value < -1 else "Similar defense"
                else:
                    context = ""
                
                feature_explanations.append({
                    'name': feat_name,
                    'label': explanation['label'],
                    'icon': explanation['icon'],
                    'value': value,
                    'description': explanation['description'],
                    'context': context
                })
        
        # Ensure all expected features are present
        X = pd.DataFrame([{feat: features.get(feat, 0) for feat in self.feature_names}])
        
        # Predict
        probs = self.model.predict_proba(X)[0]
        pred_class = self.model.predict(X)[0]
        
        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        
        # Get confidence level
        confidence = max(probs)
        if confidence > 0.7:
            confidence_level = "High Confidence"
        elif confidence > 0.5:
            confidence_level = "Medium Confidence"
        else:
            confidence_level = "Low Confidence"
        
        return {
            "outcome": outcome_map[pred_class],
            "probabilities": {
                "home_win": float(probs[0]),
                "draw": float(probs[1]),
                "away_win": float(probs[2])
            },
            "features": {k: float(v) for k, v in features.items() if k in self.feature_names},
            "feature_explanations": feature_explanations,
            "home_form_last5": home_last5,
            "away_form_last5": away_last5,
            "home_form_points": home_form,
            "away_form_points": away_form,
            "home_form_description": self.get_form_strength_description(home_form),
            "away_form_description": self.get_form_strength_description(away_form),
            "confidence_level": confidence_level,
            "venue": venue or home_club.get("stadium_name", "Unknown")
        }
    
    def calculate_form_trend(self, club_id: int) -> float:
        """Calculate if team is improving or declining"""
        games = get_games_history(club_id, limit=8, season=self.current_season)
        
        if len(games) < 6:
            games = get_games_history(club_id, limit=8, season="2024")
        
        if len(games) < 6:
            return 0.0
        
        # Points from last 3 games
        recent_points = 0
        for game in games[:3]:
            if game['goals_for'] > game['goals_against']:
                recent_points += 3
            elif game['goals_for'] == game['goals_against']:
                recent_points += 1
        
        # Points from games 3-6
        earlier_points = 0
        for game in games[3:6]:
            if game['goals_for'] > game['goals_against']:
                earlier_points += 3
            elif game['goals_for'] == game['goals_against']:
                earlier_points += 1
        
        return (recent_points - earlier_points) / 3


class CardsPredictor:
    """Yellow/red cards predictor using actual referee and club data"""
    
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        # Cache for referee stats to avoid repeated DB queries
        self._referee_cache = {}
        self._club_cache = {}
        
    def get_referee_avg_cards(self, referee_name: str) -> float:
        """Get average cards per game for a specific referee from database"""
        if referee_name in self._referee_cache:
            return self._referee_cache[referee_name]
        
        try:
            from app.database import get_db_connection
            
            with get_db_connection() as conn:
                # Get all games for this referee
                cursor = conn.execute(
                    """SELECT g.game_id, COUNT(ge.game_event_id) as card_count
                       FROM games g
                       LEFT JOIN game_events ge ON g.game_id = ge.game_id AND ge.type = 'Cards'
                       WHERE g.referee = ?
                       GROUP BY g.game_id""",
                    (referee_name,)
                )
                games_data = cursor.fetchall()
                
                if games_data:
                    total_cards = sum(row[1] for row in games_data)
                    total_games = len(games_data)
                    avg_cards = total_cards / total_games if total_games > 0 else 3.5
                else:
                    # No data for this referee, use default
                    avg_cards = 3.5
                
                self._referee_cache[referee_name] = avg_cards
                return avg_cards
                
        except Exception as e:
            print(f"Error getting referee data for {referee_name}: {e}")
            return 3.5
    
    def get_club_avg_cards(self, club_id: int, is_home: bool = True) -> float:
        """Get average cards for a club (home or away)"""
        cache_key = f"{club_id}_{is_home}"
        if cache_key in self._club_cache:
            return self._club_cache[cache_key]
        
        try:
            from app.database import get_db_connection
            
            with get_db_connection() as conn:
                if is_home:
                    cursor = conn.execute(
                        """SELECT g.game_id, COUNT(ge.game_event_id) as card_count
                           FROM games g
                           LEFT JOIN game_events ge ON g.game_id = ge.game_id AND ge.type = 'Cards'
                           WHERE g.home_club_id = ?
                           GROUP BY g.game_id""",
                        (club_id,)
                    )
                else:
                    cursor = conn.execute(
                        """SELECT g.game_id, COUNT(ge.game_event_id) as card_count
                           FROM games g
                           LEFT JOIN game_events ge ON g.game_id = ge.game_id AND ge.type = 'Cards'
                           WHERE g.away_club_id = ?
                           GROUP BY g.game_id""",
                        (club_id,)
                    )
                
                games_data = cursor.fetchall()
                
                if games_data:
                    total_cards = sum(row[1] for row in games_data)
                    total_games = len(games_data)
                    avg_cards = total_cards / total_games if total_games > 0 else 2.5
                else:
                    avg_cards = 2.5
                
                self._club_cache[cache_key] = avg_cards
                return avg_cards
                
        except Exception as e:
            print(f"Error getting club cards data: {e}")
            return 2.5
    
    def get_club_position(self, club_id: int) -> float:
        """Get current league position for a club"""
        try:
            from app.database import get_latest_position
            position = get_latest_position(club_id, "2025")
            if position is None:
                position = get_latest_position(club_id, "2024")
            return position if position else 10.0
        except Exception as e:
            print(f"Error getting position: {e}")
            return 10.0
    
    def predict(self, home_club: Dict, away_club: Dict, referee: str) -> Dict[str, Any]:
        """Predict if match will have 3+ cards using real data"""
        
        home_id = home_club.get('club_id')
        away_id = away_club.get('club_id')
        
        # Get real data from database
        ref_avg_cards = self.get_referee_avg_cards(referee)
        home_avg_cards = self.get_club_avg_cards(home_id, is_home=True)
        away_avg_cards = self.get_club_avg_cards(away_id, is_home=False)
        home_position = self.get_club_position(home_id)
        away_position = self.get_club_position(away_id)
        
        # Calculate match intensity factor (derby or high-stakes match)
        # You can enhance this based on rivalry, league importance, etc.
        intensity_factor = 1.0
        
        # Adjust predictions based on referee style
        if ref_avg_cards > 4.5:
            referee_style = "Very Strict"
        elif ref_avg_cards > 3.5:
            referee_style = "Strict"
        elif ref_avg_cards > 2.5:
            referee_style = "Moderate"
        else:
            referee_style = "Lenient"
        
        # Create features for the model
        features = pd.DataFrame([{
            'home_club_position': home_position,
            'away_club_position': away_position,
            'avg_cards_per_game_referee': ref_avg_cards,
            'avg_cards_home': home_avg_cards,
            'avg_cards_away': away_avg_cards
        }])
        
        # Get model prediction
        try:
            prob = self.model.predict_proba(features)[0][1]
        except:
            # Fallback calculation if model fails
            # Higher cards when: strict referee + high card teams + low positions (more fighting)
            prob = min(0.9, (ref_avg_cards / 5.0) * 0.5 + 
                             ((home_avg_cards + away_avg_cards) / 6.0) * 0.3 +
                             ((20 - (home_position + away_position)/2) / 20) * 0.2)
        
        # Apply intensity factor
        prob = min(0.95, prob * intensity_factor)
        
        # Generate estimate based on probability and referee style
        if prob < 0.3:
            estimate = f"Low card match (1-2 cards) - {referee_style} referee {referee}"
            risk_color = "green"
            risk_level = "Low"
        elif prob < 0.5:
            estimate = f"Moderate card match (2-3 cards) - {referee_style} referee {referee}"
            risk_color = "orange"
            risk_level = "Medium"
        elif prob < 0.7:
            estimate = f"Above average cards (3-4 cards) - {referee_style} referee {referee}"
            risk_color = "orange"
            risk_level = "Medium-High"
        elif prob < 0.85:
            estimate = f"High card match (4-5 cards) - {referee_style} referee {referee}"
            risk_color = "red"
            risk_level = "High"
        else:
            estimate = f"Very high card match (5+ cards expected) - {referee_style} referee {referee}"
            risk_color = "darkred"
            risk_level = "Very High"
        
        return {
            "probability_3plus": float(prob),
            "estimate": estimate,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "referee_style": referee_style,
            "referee_avg_cards": round(ref_avg_cards, 2),
            "home_team_avg_cards": round(home_avg_cards, 2),
            "away_team_avg_cards": round(away_avg_cards, 2)
        }