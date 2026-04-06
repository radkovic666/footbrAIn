"""
Betting strategies for VARIO system
Pre-configured strategies based on risk tolerance
"""

from typing import Dict

# Pre-configured betting strategies
BETTING_STRATEGIES = {
    'conservative': {
        'name': '🛡️ Conservative',
        'min_ev_threshold': 0.08,      # 8% minimum EV
        'kelly_fraction': 0.15,         # 15% Kelly (very conservative)
        'max_stake_percent': 0.03,      # Max 3% of bankroll
        'draw_penalty': 0.15,           # +15% EV required for draws
        'description': 'Low risk, fewer bets, suitable for beginners',
        'color': '#00ff88'
    },
    'balanced': {
        'name': '⚖️ Balanced',
        'min_ev_threshold': 0.05,       # 5% minimum EV
        'kelly_fraction': 0.25,         # 25% Kelly
        'max_stake_percent': 0.05,      # Max 5% of bankroll
        'draw_penalty': 0.10,           # +10% EV required for draws
        'description': 'Moderate risk, optimized for 70% accuracy model',
        'color': '#ffaa00'
    },
    'aggressive': {
        'name': '⚡ Aggressive',
        'min_ev_threshold': 0.03,       # 3% minimum EV
        'kelly_fraction': 0.40,         # 40% Kelly
        'max_stake_percent': 0.08,      # Max 8% of bankroll
        'draw_penalty': 0.05,           # +5% EV required for draws
        'description': 'Higher risk, more betting opportunities',
        'color': '#ff4444'
    }
}

def get_strategy(strategy_name: str = 'balanced') -> Dict:
    """Get a betting strategy configuration"""
    if strategy_name not in BETTING_STRATEGIES:
        return BETTING_STRATEGIES['balanced']
    return BETTING_STRATEGIES[strategy_name]

def get_all_strategies() -> Dict:
    """Get all available strategies"""
    return BETTING_STRATEGIES