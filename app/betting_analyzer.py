"""
Value Betting Analyzer for VARIO Football Prediction System
Calculates Expected Value (EV) and Kelly Criterion stakes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingAnalyzer:
    """Value betting analysis based on model predictions vs bookmaker odds"""
    
    def __init__(self, 
                 min_ev_threshold: float = 0.05,
                 kelly_fraction: float = 0.25,
                 max_stake_percent: float = 0.05,
                 draw_penalty: float = 0.10):
        """
        Args:
            min_ev_threshold: Minimum EV to consider a bet (default 0.05 = 5%)
            kelly_fraction: Fraction of full Kelly to use (0.25 = conservative)
            max_stake_percent: Maximum stake as % of bankroll (5%)
            draw_penalty: Extra EV required for draw bets (+10%)
        """
        self.min_ev_threshold = min_ev_threshold
        self.kelly_fraction = kelly_fraction
        self.max_stake_percent = max_stake_percent
        self.draw_penalty = draw_penalty
        
        # Outcome mapping
        self.outcomes = ['home_win', 'draw', 'away_win']
        self.outcome_display = {
            'home_win': '🏠 Home Win',
            'draw': '🤝 Draw',
            'away_win': '🚗 Away Win'
        }
        
    def calculate_ev(self, model_prob: float, odds: float) -> float:
        """Calculate Expected Value for a single outcome"""
        if odds <= 0 or model_prob <= 0:
            return -1
        return (model_prob * odds) - 1
    
    def calculate_kelly_stake(self, 
                              bankroll: float, 
                              model_prob: float, 
                              odds: float) -> float:
        """
        Calculate Kelly Criterion stake
        Formula: (odds * prob - 1) / (odds - 1)
        """
        if odds <= 1:
            return 0
        
        kelly_percent = (odds * model_prob - 1) / (odds - 1)
        
        # Apply fractional Kelly and cap
        kelly_percent = kelly_percent * self.kelly_fraction
        kelly_percent = max(0, min(kelly_percent, self.max_stake_percent))
        
        return bankroll * kelly_percent
    
    def analyze_bet(self, 
                    model_probs: Dict[str, float],
                    bookmaker_odds: Dict[str, float],
                    bankroll: float = 1000) -> Dict:
        """
        Analyze a single match for value betting opportunities
        
        Args:
            model_probs: {'home_win': 0.6, 'draw': 0.2, 'away_win': 0.2}
            bookmaker_odds: {'home_win': 1.8, 'draw': 3.8, 'away_win': 4.5}
            bankroll: Current bankroll amount
        
        Returns:
            Bet recommendation dict
        """
        results = []
        
        for outcome in self.outcomes:
            model_prob = model_probs.get(outcome, 0)
            odds = bookmaker_odds.get(outcome, 0)
            
            if model_prob <= 0 or odds <= 0:
                continue
            
            ev = self.calculate_ev(model_prob, odds)
            
            # Special handling for draws (require higher EV due to lower model accuracy)
            if outcome == 'draw':
                min_ev = self.min_ev_threshold + self.draw_penalty
            else:
                min_ev = self.min_ev_threshold
            
            is_value_bet = ev > min_ev
            stake = self.calculate_kelly_stake(bankroll, model_prob, odds) if is_value_bet else 0
            
            results.append({
                'outcome': outcome,
                'outcome_display': self.outcome_display[outcome],
                'model_probability': round(model_prob, 3),
                'bookmaker_odds': odds,
                'decimal_odds': odds,
                'expected_value': round(ev, 3),
                'expected_value_percent': round(ev * 100, 1),
                'is_value_bet': is_value_bet,
                'recommended_stake': round(stake, 2),
                'stake_percent': round((stake / bankroll) * 100, 2),
                'confidence': self._get_confidence_level(model_prob, ev)
            })
        
        # Sort by expected value (best first)
        results.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # Find best bet
        best_bet = next((r for r in results if r['is_value_bet']), None)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'best_bet': best_bet,
            'all_bets': results,
            'has_value_bet': best_bet is not None,
            'summary': self._generate_summary(results, best_bet)
        }
    
    def _get_confidence_level(self, prob: float, ev: float) -> str:
        """Determine confidence level based on probability and EV"""
        if prob > 0.65 and ev > 0.10:
            return "HIGH"
        elif prob > 0.55 and ev > 0.07:
            return "MEDIUM"
        elif ev > 0.05:
            return "LOW"
        else:
            return "NONE"
    
    def _generate_summary(self, results: List[Dict], best_bet: Dict) -> str:
        """Generate human-readable summary"""
        if not best_bet:
            return "❌ No value bets found for this match - Skip betting"
        
        return f"""
✅ VALUE BET FOUND!
Outcome: {best_bet['outcome_display']}
Model Probability: {best_bet['model_probability']*100:.1f}%
Bookmaker Odds: {best_bet['bookmaker_odds']:.2f}
Expected Value: +{best_bet['expected_value_percent']:.1f}%
Recommended Stake: ${best_bet['recommended_stake']:.2f} ({best_bet['stake_percent']:.1f}% of bankroll)
Confidence: {best_bet['confidence']}
        """.strip()
    
    def get_betting_advice(self, model_probs: Dict, bookmaker_odds: Dict, bankroll: float = 1000) -> str:
        """Get simple betting advice text"""
        analysis = self.analyze_bet(model_probs, bookmaker_odds, bankroll)
        
        if not analysis['has_value_bet']:
            return "❌ NO VALUE BET - Skip this match"
        
        best = analysis['best_bet']
        return f"✅ BET on {best['outcome_display']} - EV: +{best['expected_value_percent']:.1f}% - Stake: ${best['recommended_stake']:.2f}"


class BacktestEngine:
    """Backtest betting strategy on historical data"""
    
    def __init__(self, 
                 initial_bankroll: float = 1000,
                 min_ev_threshold: float = 0.05,
                 kelly_fraction: float = 0.25):
        
        self.initial_bankroll = initial_bankroll
        self.analyzer = BettingAnalyzer(
            min_ev_threshold=min_ev_threshold,
            kelly_fraction=kelly_fraction
        )
        self.results = []
        
    def simulate_bet(self, 
                     model_probs: Dict, 
                     bookmaker_odds: Dict, 
                     actual_result: str,
                     bankroll: float) -> Tuple[float, Dict]:
        """
        Simulate a single bet for backtesting
        
        Returns:
            (new_bankroll, bet_result_details)
        """
        analysis = self.analyzer.analyze_bet(model_probs, bookmaker_odds, bankroll)
        
        if not analysis['has_value_bet']:
            return bankroll, {'bet_placed': False, 'reason': 'no_value'}
        
        best_bet = analysis['best_bet']
        stake = best_bet['recommended_stake']
        
        # Check if bet won
        outcome_map = {
            'home_win': 'Home Win',
            'draw': 'Draw', 
            'away_win': 'Away Win'
        }
        
        bet_won = (outcome_map[best_bet['outcome']] == actual_result)
        
        if bet_won:
            profit = stake * (best_bet['bookmaker_odds'] - 1)
            new_bankroll = bankroll + profit
        else:
            profit = -stake
            new_bankroll = bankroll - stake
        
        return new_bankroll, {
            'bet_placed': True,
            'outcome_bet': best_bet['outcome'],
            'outcome_display': best_bet['outcome_display'],
            'stake': stake,
            'odds': best_bet['bookmaker_odds'],
            'won': bet_won,
            'profit': profit,
            'expected_value': best_bet['expected_value'],
            'model_probability': best_bet['model_probability']
        }
    
    def run_backtest(self, matches_data: List[Dict]) -> Dict:
        """
        Run backtest on historical matches
        
        Args:
            matches_data: List of dicts with:
                - model_probs: {'home_win': 0.6, 'draw': 0.2, 'away_win': 0.2}
                - bookmaker_odds: {'home_win': 1.8, 'draw': 3.8, 'away_win': 4.5}
                - actual_result: "Home Win" or "Draw" or "Away Win"
        
        Returns:
            Backtest statistics
        """
        bankroll = self.initial_bankroll
        bets_placed = []
        
        for match in matches_data:
            bankroll, bet_result = self.simulate_bet(
                match['model_probs'],
                match['bookmaker_odds'],
                match['actual_result'],
                bankroll
            )
            
            if bet_result.get('bet_placed'):
                bets_placed.append(bet_result)
            
            self.results.append({
                **match,
                'bet_result': bet_result,
                'bankroll_after': bankroll
            })
        
        # Calculate statistics
        total_profit = bankroll - self.initial_bankroll
        roi = (total_profit / self.initial_bankroll) * 100
        
        winning_bets = [b for b in bets_placed if b.get('won')]
        losing_bets = [b for b in bets_placed if not b.get('won')]
        
        win_rate = len(winning_bets) / len(bets_placed) * 100 if bets_placed else 0
        
        # Calculate EV efficiency
        avg_ev = np.mean([b.get('expected_value', 0) for b in bets_placed]) if bets_placed else 0
        actual_avg_return = np.mean([b.get('profit', 0) / b.get('stake', 1) for b in bets_placed]) if bets_placed else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [b.get('profit', 0) / b.get('stake', 1) for b in bets_placed]
        sharpe = np.mean(returns) / (np.std(returns) + 0.001) if returns else 0
        
        # Calculate max drawdown
        cumulative_returns = []
        current_bankroll = self.initial_bankroll
        for result in self.results:
            current_bankroll = result['bankroll_after']
            cumulative_returns.append(current_bankroll)
        
        peak = self.initial_bankroll
        max_drawdown = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        stats = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': round(bankroll, 2),
            'total_profit': round(total_profit, 2),
            'roi_percent': round(roi, 2),
            'total_bets': len(bets_placed),
            'winning_bets': len(winning_bets),
            'losing_bets': len(losing_bets),
            'win_rate_percent': round(win_rate, 2),
            'average_ev': round(avg_ev, 3),
            'actual_average_return': round(actual_avg_return, 3),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_percent': round(max_drawdown, 2),
            'profit_per_bet': round(total_profit / len(bets_placed), 2) if bets_placed else 0,
            'verdict': self._get_verdict(roi, win_rate, max_drawdown)
        }
        
        self.stats = stats
        return stats
    
    def _get_verdict(self, roi: float, win_rate: float, max_drawdown: float) -> str:
        """Generate verdict based on performance"""
        if roi > 20:
            return "EXCELLENT - Beating market significantly"
        elif roi > 10:
            return "GOOD - Profitable strategy"
        elif roi > 5:
            return "DECENT - Slight edge over market"
        elif roi > 0:
            return "MARGINAL - Small profit, needs more data"
        else:
            return "POOR - Not profitable, adjust thresholds"
    
    def generate_report(self) -> str:
        """Generate detailed backtest report"""
        stats = self.stats if hasattr(self, 'stats') else self.run_backtest([])
        
        report = f"""
{'='*60}
BACKTEST RESULTS REPORT
{'='*60}

💰 BANKROLL METRICS:
   Initial Bankroll: ${stats['initial_bankroll']:,.2f}
   Final Bankroll:   ${stats['final_bankroll']:,.2f}
   Total Profit:     ${stats['total_profit']:,.2f}
   ROI:              {stats['roi_percent']:.1f}%

📊 BETTING STATISTICS:
   Total Bets:       {stats['total_bets']}
   Winning Bets:     {stats['winning_bets']}
   Losing Bets:      {stats['losing_bets']}
   Win Rate:         {stats['win_rate_percent']:.1f}%
   Profit/Bet:       ${stats['profit_per_bet']:.2f}

📈 RISK METRICS:
   Expected Value (avg):  {stats['average_ev']:.3f}
   Actual Return (avg):   {stats['actual_average_return']:.3f}
   Sharpe Ratio:          {stats['sharpe_ratio']:.2f}
   Max Drawdown:          {stats['max_drawdown_percent']:.1f}%

🎯 VERDICT: {stats['verdict']}

{'='*60}
"""
        return report