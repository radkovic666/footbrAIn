"""
predictors.py — Match outcome predictor (v5 compatible)
=========================================================
Builds inference features that exactly mirror v5.py training features:
  - Elo ratings (replayed from full game history at startup)
  - Season-aware rolling form, win rate, draw rate, goals, GD, clean sheets
  - H2H history
  - Club-specific home advantage
  - Market value ratio
  - All composite signals (elo_closeness, draw_signal, form_closeness, etc.)

CardsPredictor is unchanged from original.
"""
import datetime
from datetime import datetime, timedelta
import joblib
import sqlite3
import pandas as pd
import numpy as np
import logging
import os
import time
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)
from config import DB_PATH, MODELS_DIR, ELO_BASE, ELO_K, ELO_HOME_ADV


class EnsembleModel:
    """Compatibility class for legacy joblib artifacts saved from v5.py."""
    def predict_proba(self, X):
        return 0.40 * self.xgb.predict_proba(X) + 0.40 * self.lgbm.predict_proba(X) + 0.20 * self.rf.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class SoftVotingArtifact:
    def __init__(self, artifact: Dict[str, Any]):
        self.xgb = artifact['xgb']
        self.lgbm = artifact['lgbm']
        self.rf = artifact['rf']
        weights = artifact.get('weights', {})
        self.w_xgb = float(weights.get('xgb', 0.40))
        self.w_lgbm = float(weights.get('lgbm', 0.40))
        self.w_rf = float(weights.get('rf', 0.20))
        self.named_estimators_ = {'xgb': self.xgb, 'lgbm': self.lgbm, 'rf': self.rf}

    def predict_proba(self, X):
        return self.w_xgb * self.xgb.predict_proba(X) + self.w_lgbm * self.lgbm.predict_proba(X) + self.w_rf * self.rf.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _load_model_artifact(model_path: str):
    import sys
    for module_name in ('__main__', '__mp_main__'):
        module = sys.modules.get(module_name)
        if module is not None and not hasattr(module, 'EnsembleModel'):
            setattr(module, 'EnsembleModel', EnsembleModel)
    model = joblib.load(model_path)
    if isinstance(model, dict) and model.get('artifact_type') == 'soft_voting_ensemble_v1':
        return SoftVotingArtifact(model)
    return model

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _active_season_window(now: Optional[datetime] = None) -> tuple[str, str, str]:
    """Return (season_key, start_date, cutoff_date) for the active football season.

    We do NOT trust the DB's season label for live inference. Instead we anchor the
    active season by real dates: Aug 1 of the start year through today.
    For April 10, 2026 this yields season_key='2025-2026', start_date='2025-08-01',
    cutoff_date='2026-04-10'.
    """
    now = now or datetime.now()
    start_year = now.year if now.month >= 8 else now.year - 1
    season_key = f"{start_year}-{start_year + 1}"
    start_date = datetime(start_year, 8, 1).date().isoformat()
    cutoff_date = now.date().isoformat()
    return season_key, start_date, cutoff_date


# ══════════════════════════════════════════════════════════════════════════════
# ELO CACHE — replays full game history once at startup
# ══════════════════════════════════════════════════════════════════════════════

class _EloCache:
    """
    Replays every game chronologically to produce current Elo for every club.
    Identical algorithm to v5.py calculate_elo(). Built once, O(1) lookup.
    """
    def __init__(self):
        self._ratings: Dict[int, float] = {}
        self._built = False

    def build(self):
        if self._built:
            return
        logger.info("Building Elo ratings from full game history...")
        t0 = time.time()
        elo: Dict[int, float] = {}

        with _db() as conn:
            rows = conn.execute("""
                SELECT home_club_id, away_club_id,
                       home_club_goals, away_club_goals
                FROM games
                WHERE home_club_goals IS NOT NULL
                  AND away_club_goals IS NOT NULL
                  AND date IS NOT NULL
                ORDER BY date ASC
            """).fetchall()

        for row in rows:
            h_id, a_id = int(row[0]), int(row[1])
            h_g,  a_g  = int(row[2]), int(row[3])
            h_elo = elo.get(h_id, ELO_BASE)
            a_elo = elo.get(a_id, ELO_BASE)

            exp_h = 1 / (1 + 10 ** ((a_elo - h_elo - ELO_HOME_ADV) / 400))
            actual_h = 1.0 if h_g > a_g else (0.5 if h_g == a_g else 0.0)

            elo[h_id] = h_elo + ELO_K * (actual_h - exp_h)
            elo[a_id] = a_elo + ELO_K * ((1 - actual_h) - (1 - exp_h))

        self._ratings = elo
        self._built = True
        logger.info(f"Elo built for {len(elo):,} clubs in {time.time()-t0:.1f}s")

    def get(self, club_id: int) -> float:
        return self._ratings.get(int(club_id), ELO_BASE)

    def win_prob(self, home_id: int, away_id: int) -> float:
        h = self.get(home_id)
        a = self.get(away_id)
        return 1 / (1 + 10 ** ((a - h - ELO_HOME_ADV) / 400))


_elo_cache = _EloCache()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_club_rolling_stats(club_id: int, season_key: Optional[str] = None, start_date: Optional[str] = None, cutoff_date: Optional[str] = None) -> Dict[str, float]:
    """
    Fetch the last 10 *played* matches for a club inside the active real-world season
    window first, then fall back to the latest 10 played matches before the cutoff.
    This keeps live inference aligned with the actual 2025-2026 campaign instead of
    relying on whatever text lives in the DB's `season` column.
    """
    if season_key is None or start_date is None or cutoff_date is None:
        season_key, start_date, cutoff_date = _active_season_window()

    with _db() as conn:
        rows = conn.execute("""
            SELECT
              CASE WHEN home_club_id=? THEN home_club_goals ELSE away_club_goals END as scored,
              CASE WHEN home_club_id=? THEN away_club_goals ELSE home_club_goals END as conceded
            FROM games
            WHERE (home_club_id=? OR away_club_id=?)
              AND home_club_goals IS NOT NULL
              AND away_club_goals IS NOT NULL
              AND date(date) >= date(?)
              AND date(date) <= date(?)
            ORDER BY date DESC LIMIT 10
        """, (club_id, club_id, club_id, club_id, start_date, cutoff_date)).fetchall()

    if len(rows) < 3:
        with _db() as conn:
            rows = conn.execute("""
                SELECT
                  CASE WHEN home_club_id=? THEN home_club_goals ELSE away_club_goals END,
                  CASE WHEN home_club_id=? THEN away_club_goals ELSE home_club_goals END
                FROM games
                WHERE (home_club_id=? OR away_club_id=?)
                  AND home_club_goals IS NOT NULL
                  AND away_club_goals IS NOT NULL
                  AND date(date) <= date(?)
                ORDER BY date DESC LIMIT 10
            """, (club_id, club_id, club_id, club_id, cutoff_date)).fetchall()

    if not rows:
        return _default_rolling()

    games = []
    for r in rows:
        sc, co = int(r[0] or 0), int(r[1] or 0)
        pts = 3 if sc > co else (1 if sc == co else 0)
        games.append({
            'scored': sc, 'conceded': co,
            'points': pts, 'gd': sc - co,
            'win':   1 if sc > co else 0,
            'draw':  1 if sc == co else 0,
            'clean': 1 if co == 0 else 0,
        })

    def avg(key, n):
        v = [g[key] for g in games[:n]]
        return sum(v) / len(v) if v else 0.0

    def std(key, n):
        v = [g[key] for g in games[:n]]
        if len(v) < 2:
            return 0.0
        m = sum(v) / len(v)
        return (sum((x - m)**2 for x in v) / len(v)) ** 0.5

    n5, n10 = min(5, len(games)), min(10, len(games))

    if len(games) >= 6:
        trend = sum(g['points'] for g in games[:3]) / 3 - sum(g['points'] for g in games[3:6]) / 3
    else:
        trend = 0.0

    return {
        'form5':        avg('points', n5),
        'form10':       avg('points', n10),
        'win_rate5':    avg('win',    n5),
        'win_rate10':   avg('win',    n10),
        'draw_rate10':  avg('draw',   n10),
        'scored5':      avg('scored', n5),
        'conceded5':    avg('conceded', n5),
        'gd5':          avg('gd',     n5),
        'gd10':         avg('gd',     n10),
        'clean5':       float(sum(g['clean'] for g in games[:n5])),
        'scored_std5':  std('scored', n5),
        'form_trend':   trend,
    }


def _default_rolling() -> Dict[str, float]:
    return {
        'form5': 1.2, 'form10': 1.2,
        'win_rate5': 0.40, 'win_rate10': 0.40,
        'draw_rate10': 0.25,
        'scored5': 1.3, 'conceded5': 1.2,
        'gd5': 0.1, 'gd10': 0.1,
        'clean5': 1.0, 'scored_std5': 0.8,
        'form_trend': 0.0,
    }


def _get_coach_stats(manager_name: str) -> tuple:
    """
    Return (bayesian_ppg, career_games, is_new_coach) for a manager
    from the full game history. Bayesian prior = 1.25 ppg over 10 games.
    """
    if not manager_name or manager_name.strip() == '':
        return (1.25, 0, 1)

    PRIOR_PPG    = 1.25
    PRIOR_WEIGHT = 10
    NEW_THRESH   = 8

    with _db() as conn:
        rows = conn.execute("""
            SELECT
              SUM(CASE
                WHEN home_club_manager_name=? AND home_club_goals > away_club_goals THEN 3
                WHEN home_club_manager_name=? AND home_club_goals = away_club_goals THEN 1
                WHEN home_club_manager_name=? AND home_club_goals < away_club_goals THEN 0
                WHEN away_club_manager_name=? AND away_club_goals > home_club_goals THEN 3
                WHEN away_club_manager_name=? AND away_club_goals = home_club_goals THEN 1
                WHEN away_club_manager_name=? AND away_club_goals < home_club_goals THEN 0
                ELSE 0 END) as pts,
              COUNT(*) as games
            FROM games
            WHERE (home_club_manager_name=? OR away_club_manager_name=?)
              AND home_club_goals IS NOT NULL
        """, (manager_name,)*8).fetchone()

    pts   = int(rows[0] or 0)
    games = int(rows[1] or 0)
    ppg   = (pts + PRIOR_PPG * PRIOR_WEIGHT) / (games + PRIOR_WEIGHT)
    is_new = 1 if games < NEW_THRESH else 0
    return (ppg, games, is_new)


def _get_h2h_stats(home_id: int, away_id: int, cutoff_date: Optional[str] = None) -> Dict[str, float]:
    cutoff_date = cutoff_date or _active_season_window()[2]
    with _db() as conn:
        rows = conn.execute("""
            SELECT home_club_id, home_club_goals, away_club_goals
            FROM games
            WHERE ((home_club_id=? AND away_club_id=?)
                OR (home_club_id=? AND away_club_id=?))
              AND home_club_goals IS NOT NULL
              AND away_club_goals IS NOT NULL
              AND date(date) <= date(?)
            ORDER BY date DESC LIMIT 6
        """, (home_id, away_id, away_id, home_id, cutoff_date)).fetchall()

    if not rows:
        return {'h2h_draw_rate': 0.25, 'h2h_home_win_rate': 0.45,
                'h2h_avg_goals': 2.5,  'h2h_n': 0}

    n = len(rows)
    draws = hw = goals = 0
    for r in rows:
        hg, ag = int(r[1] or 0), int(r[2] or 0)
        goals += hg + ag
        if hg == ag:
            draws += 1
        elif int(r[0]) == home_id and hg > ag:
            hw += 1
        elif int(r[0]) == away_id and ag > hg:
            hw += 1

    return {
        'h2h_draw_rate':     draws / n,
        'h2h_home_win_rate': hw / n,
        'h2h_avg_goals':     goals / n,
        'h2h_n':             float(n),
    }


def _get_h2h_matches(home_id: int, away_id: int, limit: int = 5, cutoff_date: Optional[str] = None) -> List[Dict[str, Any]]:
    cutoff_date = cutoff_date or _active_season_window()[2]
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT date, home_club_id, away_club_id, home_club_name, away_club_name,
                   home_club_goals, away_club_goals, competition_id
            FROM games
            WHERE ((home_club_id=? AND away_club_id=?) OR (home_club_id=? AND away_club_id=?))
              AND home_club_goals IS NOT NULL
              AND away_club_goals IS NOT NULL
              AND date(date) <= date(?)
            ORDER BY date DESC LIMIT ?
            """,
            (home_id, away_id, away_id, home_id, cutoff_date, limit),
        ).fetchall()
    out=[]
    for r in rows:
        date, h_id, a_id, h_name, a_name, hg, ag, comp_id = r
        hg=int(hg or 0); ag=int(ag or 0)
        if hg == ag:
            res='D'
        elif int(h_id)==home_id and hg>ag:
            res='W'
        elif int(a_id)==home_id and ag>hg:
            res='W'
        else:
            res='L'
        out.append({'date':date,'home_team':h_name,'away_team':a_name,'score':f"{hg}-{ag}",'competition_id':comp_id,'result_for_home_team':res})
    return out


def _get_club_home_win_rate(club_id: int, comp_id=None, cutoff_date: Optional[str] = None) -> float:
    cutoff_date = cutoff_date or _active_season_window()[2]
    with _db() as conn:
        if comp_id:
            r = conn.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN home_club_goals > away_club_goals THEN 1 ELSE 0 END)
                FROM games
                WHERE home_club_id=? AND competition_id=?
                  AND home_club_goals IS NOT NULL AND away_club_goals IS NOT NULL
                  AND date(date) <= date(?)
            """, (club_id, comp_id, cutoff_date)).fetchone()
        else:
            r = conn.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN home_club_goals > away_club_goals THEN 1 ELSE 0 END)
                FROM games
                WHERE home_club_id=? AND home_club_goals IS NOT NULL AND away_club_goals IS NOT NULL
                  AND date(date) <= date(?)
            """, (club_id, cutoff_date)).fetchone()
    total, wins = int(r[0] or 0), int(r[1] or 0)
    return wins / total if total >= 5 else 0.48


def _get_team_value(club_id: int, cutoff_date: Optional[str] = None) -> float:
    cutoff_date = cutoff_date or _active_season_window()[2]
    with _db() as conn:
        r = conn.execute("""
            SELECT SUM(pv.market_value_in_eur)
            FROM player_valuations pv
            INNER JOIN (
                SELECT player_id, MAX(date) as md
                FROM player_valuations
                WHERE current_club_id=? AND date(date) <= date(?)
                GROUP BY player_id
            ) lv ON pv.player_id=lv.player_id AND pv.date=lv.md
            WHERE pv.current_club_id=?
        """, (club_id, cutoff_date, club_id)).fetchone()
    return float(r[0] or 0)


def _get_last5_display(club_id: int, season_key: Optional[str] = None, start_date: Optional[str] = None, cutoff_date: Optional[str] = None) -> List[Dict]:
    if season_key is None or start_date is None or cutoff_date is None:
        season_key, start_date, cutoff_date = _active_season_window()

    with _db() as conn:
        rows = conn.execute("""
            SELECT g.date,
              CASE WHEN g.home_club_id=? THEN g.home_club_goals ELSE g.away_club_goals END as gf,
              CASE WHEN g.home_club_id=? THEN g.away_club_goals ELSE g.home_club_goals END as ga,
              CASE WHEN g.home_club_id=? THEN g.away_club_name ELSE g.home_club_name END as opp,
              CASE WHEN g.home_club_id=? THEN 1 ELSE 0 END as ih,
              g.season
            FROM games g
            WHERE (g.home_club_id=? OR g.away_club_id=?)
              AND g.home_club_goals IS NOT NULL AND g.away_club_goals IS NOT NULL
              AND date(g.date) >= date(?)
              AND date(g.date) <= date(?)
            ORDER BY g.date DESC LIMIT 5
        """, (club_id, club_id, club_id, club_id, club_id, club_id, start_date, cutoff_date)).fetchall()

    if len(rows) < 3:
        with _db() as conn:
            rows = conn.execute("""
                SELECT g.date,
                  CASE WHEN g.home_club_id=? THEN g.home_club_goals ELSE g.away_club_goals END as gf,
                  CASE WHEN g.home_club_id=? THEN g.away_club_goals ELSE g.home_club_goals END as ga,
                  CASE WHEN g.home_club_id=? THEN g.away_club_name ELSE g.home_club_name END as opp,
                  CASE WHEN g.home_club_id=? THEN 1 ELSE 0 END as ih,
                  g.season
                FROM games g
                WHERE (g.home_club_id=? OR g.away_club_id=?)
                  AND g.home_club_goals IS NOT NULL AND g.away_club_goals IS NOT NULL
                  AND date(g.date) <= date(?)
                ORDER BY g.date DESC LIMIT 5
            """, (club_id, club_id, club_id, club_id, club_id, club_id, cutoff_date)).fetchall()

    cm = {'W': 'green', 'D': 'orange', 'L': 'red'}
    out = []
    for r in rows:
        gf, ga = int(r[1] or 0), int(r[2] or 0)
        res = 'W' if gf > ga else ('D' if gf == ga else 'L')
        out.append({
            'date': r[0], 'goals_for': gf, 'goals_against': ga,
            'opponent': r[3], 'is_home': bool(r[4]), 'season': r[5],
            'result': res, 'result_color': cm[res],
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class MatchPredictor:
    """
    Match outcome predictor — v5 model compatible.
    Computes the exact same features as v5.py training so the model
    receives correctly scaled, correctly named inputs.
    """

    def __init__(self, model_path: str, features_path: str,
                 model_version: str = "v5"):
        self.model          = _load_model_artifact(model_path)
        feature_info        = joblib.load(features_path)
        self.feature_names  = feature_info.get('feature_names', [])
        self.model_version  = model_version
        self.model_type     = feature_info.get('model_type', 'unknown')
        self.accuracy       = feature_info.get('accuracy', 0.0)
        self.draw_threshold = feature_info.get('draw_threshold', 0.5)
        self.country = feature_info.get('country')
        self.league_name = feature_info.get('league_name')
        self.competition_id = feature_info.get('competition_id')
        self.teams = feature_info.get('teams', [])
        thresholds_path = features_path.replace('_features.pkl', '_thresholds.pkl')
        if not os.path.exists(thresholds_path):
            thresholds_path = os.path.join(MODELS_DIR, 'match_outcome_model_v5_thresholds.pkl')
        if os.path.exists(thresholds_path):
            try:
                threshold_info = joblib.load(thresholds_path)
                self.draw_threshold = float(threshold_info.get('draw_threshold', self.draw_threshold))
            except Exception:
                pass
        self.current_season, self.season_start_date, self.reference_cutoff_date = _active_season_window()

        # v5 stacking model handles scaling internally — no external scaler
        self.scaler   = None
        self.selector = None

        # Build Elo once at startup
        _elo_cache.build()

        logger.info(
            f"MatchPredictor {model_version} loaded | "
            f"{len(self.feature_names)} features | "
            f"accuracy={self.accuracy*100:.1f}% | "
            f"draw_thresh={self.draw_threshold:.2f}"
        )

    def get_model_info(self) -> Dict:
        return {
            'version':       self.model_version,
            'type':          self.model_type,
            'accuracy':      self.accuracy,
            'feature_count': len(self.feature_names),
            'features':      self.feature_names,
            'country':       self.country,
            'league_name':   self.league_name,
            'competition_id': self.competition_id,
        }

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def predict(self, home_club: Dict, away_club: Dict,
                referee=None, venue=None) -> Dict[str, Any]:

        home_id = int(home_club['club_id'])
        away_id = int(away_club['club_id'])

        # ── Elo ──────────────────────────────────────────────────────────────
        home_elo      = _elo_cache.get(home_id)
        away_elo      = _elo_cache.get(away_id)
        elo_diff      = home_elo - away_elo
        elo_diff_norm = elo_diff / 400
        elo_win_prob  = _elo_cache.win_prob(home_id, away_id)
        elo_superiority = float(np.tanh(elo_diff / 200))
        elo_closeness = 1.0 / (abs(elo_diff) + 1)

        # ── Rolling stats ────────────────────────────────────────────────────
        season_key, start_date, cutoff_date = _active_season_window()
        self.current_season = season_key
        self.season_start_date = start_date
        self.reference_cutoff_date = cutoff_date
        hs  = _get_club_rolling_stats(home_id, season_key, start_date, cutoff_date)
        aws = _get_club_rolling_stats(away_id, season_key, start_date, cutoff_date)

        # ── Composites ───────────────────────────────────────────────────────
        form_diff5        = hs['form5']     - aws['form5']
        form_diff10       = hs['form10']    - aws['form10']
        relative_strength = (hs['form5'] + 0.1) / (aws['form5'] + 0.1)
        win_rate_diff5    = hs['win_rate5'] - aws['win_rate5']
        win_rate_diff10   = hs['win_rate10']- aws['win_rate10']
        trend_diff        = hs['form_trend']- aws['form_trend']
        gd_diff5          = hs['gd5']       - aws['gd5']
        gd_diff10         = hs['gd10']      - aws['gd10']
        attack_diff       = hs['scored5']   - aws['scored5']
        defense_diff      = hs['conceded5'] - aws['conceded5']
        home_attack_def   = (hs['scored5']  + 0.1) / (hs['conceded5']  + 0.1)
        away_attack_def   = (aws['scored5'] + 0.1) / (aws['conceded5'] + 0.1)
        attack_def_diff   = home_attack_def - away_attack_def
        exp_total_goals   = hs['scored5'] + aws['scored5']
        combined_draw_rate= (hs['draw_rate10'] + aws['draw_rate10']) / 2
        form_closeness    = 1.0 / (abs(form_diff5) + 0.1)
        draw_signal       = combined_draw_rate * elo_closeness

        # ── H2H ─────────────────────────────────────────────────────────────
        h2h = _get_h2h_stats(home_id, away_id, cutoff_date)
        h2h_matches = _get_h2h_matches(home_id, away_id, limit=5, cutoff_date=cutoff_date)

        # ── Coach features ────────────────────────────────────────────────────
        home_coach_ppg, home_coach_games, home_coach_new = _get_coach_stats(
            home_club.get('coach_name') or home_club.get('home_club_manager_name') or '')
        away_coach_ppg, away_coach_games, away_coach_new = _get_coach_stats(
            away_club.get('coach_name') or away_club.get('away_club_manager_name') or '')
        coach_ppg_diff  = home_coach_ppg - away_coach_ppg
        home_coach_exp  = float(np.log1p(home_coach_games))
        away_coach_exp  = float(np.log1p(away_coach_games))
        coach_exp_diff  = home_coach_exp - away_coach_exp

        # ── Home advantage ───────────────────────────────────────────────────
        comp_id = home_club.get('domestic_competition_id')
        club_home_win_rate = _get_club_home_win_rate(home_id, comp_id, cutoff_date)

        # ── Market values ────────────────────────────────────────────────────
        hv = _get_team_value(home_id, cutoff_date)
        av = _get_team_value(away_id, cutoff_date)
        value_ratio     = (hv + 1e6) / (av + 1e6)
        value_diff_norm = (hv - av) / (hv + av + 1e6)

        # ── Full feature dict ────────────────────────────────────────────────
        feats = {
            'home_elo': home_elo,      'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_win_prob': elo_win_prob, 'elo_superiority': elo_superiority,
            'elo_closeness': elo_closeness,
            'home_form5': hs['form5'],   'away_form5': aws['form5'],
            'home_form10': hs['form10'], 'away_form10': aws['form10'],
            'form_diff5': form_diff5,    'form_diff10': form_diff10,
            'relative_strength': relative_strength,
            'home_form_trend': hs['form_trend'], 'away_form_trend': aws['form_trend'],
            'trend_diff': trend_diff,
            'home_win_rate5': hs['win_rate5'],  'away_win_rate5': aws['win_rate5'],
            'home_win_rate10': hs['win_rate10'],'away_win_rate10': aws['win_rate10'],
            'win_rate_diff5': win_rate_diff5,   'win_rate_diff10': win_rate_diff10,
            'home_scored5': hs['scored5'],   'away_scored5': aws['scored5'],
            'home_conceded5': hs['conceded5'],'away_conceded5': aws['conceded5'],
            'home_gd5': hs['gd5'],  'away_gd5': aws['gd5'],
            'home_gd10': hs['gd10'],'away_gd10': aws['gd10'],
            'gd_diff5': gd_diff5,   'gd_diff10': gd_diff10,
            'attack_diff': attack_diff,   'defense_diff': defense_diff,
            'home_attack_def': home_attack_def,'away_attack_def': away_attack_def,
            'attack_def_diff': attack_def_diff,'exp_total_goals': exp_total_goals,
            'home_clean5': hs['clean5'],  'away_clean5': aws['clean5'],
            'home_scored_std5': hs['scored_std5'],'away_scored_std5': aws['scored_std5'],
            'home_draw_rate10': hs['draw_rate10'],'away_draw_rate10': aws['draw_rate10'],
            'combined_draw_rate': combined_draw_rate,
            'form_closeness': form_closeness,'draw_signal': draw_signal,
            'club_home_win_rate': club_home_win_rate,
            'value_ratio': value_ratio,'value_diff_norm': value_diff_norm,
            'h2h_draw_rate': h2h['h2h_draw_rate'],
            'h2h_home_win_rate': h2h['h2h_home_win_rate'],
            'h2h_avg_goals': h2h['h2h_avg_goals'],
            'h2h_n': h2h['h2h_n'],
            'is_league': 1.0,
            # Coach
            'home_coach_ppg': home_coach_ppg,
            'away_coach_ppg': away_coach_ppg,
            'coach_ppg_diff': coach_ppg_diff,
            'home_coach_new': float(home_coach_new),
            'away_coach_new': float(away_coach_new),
            'home_coach_sgames': float(min(home_coach_games, 38)),
            'away_coach_sgames': float(min(away_coach_games, 38)),
            'home_coach_exp': home_coach_exp,
            'away_coach_exp': away_coach_exp,
            'coach_exp_diff': coach_exp_diff,
        }

        # Build input using only features the model was trained on
        X = pd.DataFrame([{f: feats.get(f, 0.0) for f in self.feature_names}])

        # ── Predict ──────────────────────────────────────────────────────────
        probs = self.model.predict_proba(X)[0]

        # Apply the same draw threshold used during training
        if probs[1] >= self.draw_threshold:
            pred_class = 1
        else:
            pred_class = 0 if probs[0] >= probs[2] else 2

        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        confidence  = float(max(probs))
        if confidence > 0.60:   conf_level = "High Confidence"
        elif confidence > 0.45: conf_level = "Medium Confidence"
        else:                   conf_level = "Low Confidence"

        home_last5 = _get_last5_display(home_id, season_key, start_date, cutoff_date)
        away_last5 = _get_last5_display(away_id, season_key, start_date, cutoff_date)

        def form_desc(f):
            if f >= 2.5: return "Excellent"
            if f >= 2.0: return "Good"
            if f >= 1.5: return "Decent"
            if f >= 1.0: return "Average"
            return "Poor"

        return {
            "model_version":         self.model_version,
            "model_accuracy":        self.accuracy,
            "outcome":               outcome_map[pred_class],
            "probabilities": {
                "home_win": float(probs[0]),
                "draw":     float(probs[1]),
                "away_win": float(probs[2]),
            },
            "features":              {k: float(v) for k, v in feats.items()
                                      if k in self.feature_names},
            "home_form_last5":       home_last5,
            "away_form_last5":       away_last5,
            "h2h_matches":           h2h_matches,
            "season_window":         {"season": season_key, "start": start_date, "cutoff": cutoff_date},
            "country":               self.country,
            "league_name":           self.league_name,
            "competition_id":        self.competition_id,
            "home_form_points":      hs['form5'],
            "away_form_points":      aws['form5'],
            "home_form_description": form_desc(hs['form5']),
            "away_form_description": form_desc(aws['form5']),
            "home_elo":              round(home_elo),
            "away_elo":              round(away_elo),
            "elo_diff":              round(elo_diff),
            "confidence_level":      conf_level,
            "venue": venue or home_club.get("stadium_name", "Unknown"),
            "season_key": season_key,
            "season_start_date": start_date,
            "reference_cutoff_date": cutoff_date,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CARDS PREDICTOR  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class CardsPredictor:
    """Yellow/red cards predictor with pre-computed statistics."""

    _referee_stats_cache   = None
    _home_club_stats_cache = None
    _away_club_stats_cache = None
    _club_stats_cache      = None
    _cache_timestamp       = None

    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            scaler_path = model_path.replace('events_outcome_model.pkl', 'cards_scaler.pkl')
            self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            self._precompute_statistics()
            logger.info(f"CardsPredictor initialised from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load cards model: {e}")
            raise

    def _precompute_statistics(self):
        try:
            t0 = time.time()
            with _db() as conn:
                rows = conn.execute("""
                    SELECT g.referee, COUNT(DISTINCT g.game_id), COUNT(ge.game_event_id)
                    FROM games g
                    LEFT JOIN game_events ge ON g.game_id=ge.game_id AND ge.type='Cards'
                    WHERE g.referee IS NOT NULL AND g.referee!=''
                    GROUP BY g.referee
                """).fetchall()
            self._referee_stats_cache = {
                r[0]: {'avg_cards': (r[2] or 0)/r[1] if r[1] else 3.5,
                        'total_games': r[1], 'total_cards': r[2] or 0}
                for r in rows
            }

            with _db() as conn:
                h_rows = conn.execute("""
                    SELECT g.home_club_id, COUNT(DISTINCT g.game_id), COUNT(ge.game_event_id)
                    FROM games g
                    LEFT JOIN game_events ge ON g.game_id=ge.game_id AND ge.type='Cards'
                    WHERE g.home_club_id IS NOT NULL GROUP BY g.home_club_id
                """).fetchall()
            self._home_club_stats_cache = {
                r[0]: (r[2] or 0)/r[1] if r[1] else 2.5 for r in h_rows
            }

            with _db() as conn:
                a_rows = conn.execute("""
                    SELECT g.away_club_id, COUNT(DISTINCT g.game_id), COUNT(ge.game_event_id)
                    FROM games g
                    LEFT JOIN game_events ge ON g.game_id=ge.game_id AND ge.type='Cards'
                    WHERE g.away_club_id IS NOT NULL GROUP BY g.away_club_id
                """).fetchall()
            self._away_club_stats_cache = {
                r[0]: (r[2] or 0)/r[1] if r[1] else 2.5 for r in a_rows
            }
            logger.info(f"Cards stats pre-computed in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed to pre-compute card stats: {e}")
            self._referee_stats_cache = {}
            self._home_club_stats_cache = {}
            self._away_club_stats_cache = {}

    def get_referee_avg_cards(self, referee: str) -> float:
        c = self._referee_stats_cache or {}
        return c[referee]['avg_cards'] if referee in c else 3.5

    def get_club_avg_cards(self, club_id: int, is_home: bool = True) -> float:
        c = self._home_club_stats_cache if is_home else self._away_club_stats_cache
        return (c or {}).get(club_id, 2.5)

    def get_club_position(self, club_id: int) -> float:
        try:
            with _db() as conn:
                r = conn.execute("""
                    SELECT home_club_position FROM games
                    WHERE home_club_id=? AND home_club_position IS NOT NULL AND season='2025'
                    ORDER BY date DESC LIMIT 1
                """, (club_id,)).fetchone()
                if r and r[0]: return float(r[0])
                r = conn.execute("""
                    SELECT away_club_position FROM games
                    WHERE away_club_id=? AND away_club_position IS NOT NULL AND season='2025'
                    ORDER BY date DESC LIMIT 1
                """, (club_id,)).fetchone()
                if r and r[0]: return float(r[0])
            return 10.0
        except Exception:
            return 10.0

    def predict(self, home_club: Dict, away_club: Dict, referee: str) -> Dict[str, Any]:
        home_id = home_club.get('club_id')
        away_id = away_club.get('club_id')

        if not referee or not referee.strip():
            return {"probability_3plus": 0.5, "estimate": "No referee specified",
                    "risk_level": "Unknown", "risk_color": "gray",
                    "referee_style": "Unknown", "referee_avg_cards": 0,
                    "home_team_avg_cards": 0, "away_team_avg_cards": 0}

        ref_avg  = self.get_referee_avg_cards(referee)
        home_avg = self.get_club_avg_cards(home_id, True)
        away_avg = self.get_club_avg_cards(away_id, False)
        home_pos = self.get_club_position(home_id)
        away_pos = self.get_club_position(away_id)

        if ref_avg > 4.5:   style = "Very Strict"
        elif ref_avg > 3.5: style = "Strict"
        elif ref_avg > 2.5: style = "Moderate"
        else:               style = "Lenient"

        features = np.array([[home_pos, away_pos, ref_avg, home_avg, away_avg]])
        if self.scaler:
            features = self.scaler.transform(features)

        try:
            prob = float(self.model.predict_proba(features)[0][1])
        except Exception as e:
            logger.error(f"Cards prediction failed: {e}")
            prob = min(0.9, (ref_avg/5)*0.5 + ((home_avg+away_avg)/6)*0.3 +
                            ((20-(home_pos+away_pos)/2)/20)*0.2)

        prob = min(0.95, max(0.05, prob))

        if prob < 0.3:   est, col, lvl = f"Low (1-2) — {style}", "green", "Low"
        elif prob < 0.5: est, col, lvl = f"Moderate (2-3) — {style}", "orange", "Medium"
        elif prob < 0.7: est, col, lvl = f"Above avg (3-4) — {style}", "orange", "Medium-High"
        elif prob < 0.85:est, col, lvl = f"High (4-5) — {style}", "red", "High"
        else:            est, col, lvl = f"Very high (5+) — {style}", "darkred", "Very High"

        return {
            "probability_3plus":   prob,
            "estimate":            est,
            "risk_level":          lvl,
            "risk_color":          col,
            "referee_style":       style,
            "referee_avg_cards":   round(ref_avg, 2),
            "home_team_avg_cards": round(home_avg, 2),
            "away_team_avg_cards": round(away_avg, 2),
        }
