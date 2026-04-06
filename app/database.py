import sqlite3
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
import os
from functools import lru_cache
from datetime import datetime

DB_PATH = "/var/www/footbrain/football_data.db"


@contextmanager
def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_referee_stats(referee_name: str) -> Dict[str, Any]:
    """Get detailed statistics for a referee"""
    with get_db_connection() as conn:
        # Get all games for this referee
        cursor = conn.execute(
            """SELECT g.game_id, g.home_club_name, g.away_club_name, g.date,
                      COUNT(ge.game_event_id) as card_count
               FROM games g
               LEFT JOIN game_events ge ON g.game_id = ge.game_id AND ge.type = 'Cards'
               WHERE g.referee = ? AND g.date IS NOT NULL
               GROUP BY g.game_id
               ORDER BY g.date DESC""",
            (referee_name,)
        )
        games = cursor.fetchall()
        
        if not games:
            return {"total_games": 0, "avg_cards": 3.5, "games": []}
        
        total_cards = sum(row[3] for row in games)
        total_games = len(games)
        avg_cards = total_cards / total_games if total_games > 0 else 3.5
        
        games_list = []
        for game in games[:10]:  # Last 10 games
            games_list.append({
                "home": game[1],
                "away": game[2],
                "date": game[3],
                "cards": game[4]
            })
        
        return {
            "total_games": total_games,
            "avg_cards": round(avg_cards, 2),
            "total_cards": total_cards,
            "recent_games": games_list
        }

def get_club_info(club_name: str) -> Optional[Dict[str, Any]]:
    """Get club information from database"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT c.club_id, c.name, c.squad_size, c.average_age, 
                      c.national_team_players, c.stadium_name, c.stadium_seats,
                      c.domestic_competition_id, comp.country_name
               FROM clubs c
               LEFT JOIN competitions comp ON c.domestic_competition_id = comp.competition_id
               WHERE c.name = ?""",
            (club_name,)
        )
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None


def get_latest_position(club_id: int, season: str = "2025") -> Optional[float]:
    """Get latest league position for a club in current season"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT home_club_position 
               FROM games 
               WHERE home_club_id = ? AND home_club_position IS NOT NULL AND season = ?
               ORDER BY date DESC LIMIT 1""",
            (club_id, season)
        )
        row = cursor.fetchone()
        
        if row and row[0]:
            return float(row[0])
        
        cursor = conn.execute(
            """SELECT away_club_position 
               FROM games 
               WHERE away_club_id = ? AND away_club_position IS NOT NULL AND season = ?
               ORDER BY date DESC LIMIT 1""",
            (club_id, season)
        )
        row = cursor.fetchone()
        
        if row and row[0]:
            return float(row[0])
        
        # Fallback to any season
        cursor = conn.execute(
            """SELECT home_club_position 
               FROM games 
               WHERE home_club_id = ? AND home_club_position IS NOT NULL
               ORDER BY date DESC LIMIT 1""",
            (club_id,)
        )
        row = cursor.fetchone()
        
        return float(row[0]) if row and row[0] else 10.0


def get_last_5_games_with_details(club_id: int, season: str = "2025") -> List[Dict]:
    """Get last 5 games with opponent and score details"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT g.date, g.home_club_goals, g.away_club_goals,
                      g.home_club_id, g.away_club_id, g.season,
                      hc.name as home_club_name, ac.name as away_club_name
               FROM games g
               LEFT JOIN clubs hc ON g.home_club_id = hc.club_id
               LEFT JOIN clubs ac ON g.away_club_id = ac.club_id
               WHERE (g.home_club_id = ? OR g.away_club_id = ?) AND g.season = ?
               ORDER BY g.date DESC LIMIT 5""",
            (club_id, club_id, season)
        )
        rows = cursor.fetchall()
        
        if len(rows) < 3:
            cursor = conn.execute(
                """SELECT g.date, g.home_club_goals, g.away_club_goals,
                          g.home_club_id, g.away_club_id, g.season,
                          hc.name as home_club_name, ac.name as away_club_name
                   FROM games g
                   LEFT JOIN clubs hc ON g.home_club_id = hc.club_id
                   LEFT JOIN clubs ac ON g.away_club_id = ac.club_id
                   WHERE (g.home_club_id = ? OR g.away_club_id = ?)
                   ORDER BY g.date DESC LIMIT 5""",
                (club_id, club_id)
            )
            rows = cursor.fetchall()
        
        games = []
        for row in rows:
            is_home = row[3] == club_id
            opponent = row[6] if is_home else row[7]
            games.append({
                "date": row[0],
                "goals_for": row[1] if is_home else row[2],
                "goals_against": row[2] if is_home else row[1],
                "is_home": is_home,
                "opponent": opponent,
                "season": row[5]
            })
        return games


def get_games_history(club_id: int, limit: int = 10, season: str = "2025") -> List[Dict]:
    """Get recent games for a club"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT date, home_club_goals, away_club_goals,
                      home_club_id, away_club_id, season
               FROM games 
               WHERE (home_club_id = ? OR away_club_id = ?) AND season = ?
               ORDER BY date DESC LIMIT ?""",
            (club_id, club_id, season, limit)
        )
        rows = cursor.fetchall()
        
        if len(rows) >= 3:
            games = []
            for row in rows:
                is_home = row[3] == club_id
                games.append({
                    "date": row[0],
                    "goals_for": row[1] if is_home else row[2],
                    "goals_against": row[2] if is_home else row[1],
                    "is_home": is_home,
                    "season": row[5]
                })
            return games
        
        # Fallback to all seasons
        cursor = conn.execute(
            """SELECT date, home_club_goals, away_club_goals,
                      home_club_id, away_club_id, season
               FROM games 
               WHERE (home_club_id = ? OR away_club_id = ?)
               ORDER BY date DESC LIMIT ?""",
            (club_id, club_id, limit)
        )
        rows = cursor.fetchall()
        
        games = []
        for row in rows:
            is_home = row[3] == club_id
            games.append({
                "date": row[0],
                "goals_for": row[1] if is_home else row[2],
                "goals_against": row[2] if is_home else row[1],
                "is_home": is_home,
                "season": row[5]
            })
        return games


@lru_cache(maxsize=1)
def get_teams_by_country() -> Dict[str, List[str]]:
    """Get all teams organized by country"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT DISTINCT c.name, comp.country_name 
               FROM clubs c
               LEFT JOIN competitions comp ON c.domestic_competition_id = comp.competition_id
               WHERE c.name IS NOT NULL AND c.name != ''
               ORDER BY comp.country_name, c.name"""
        )
        rows = cursor.fetchall()
        
        teams_by_country = {}
        for row in rows:
            country = row[1] if row[1] else "Other"
            team_name = row[0]
            if country not in teams_by_country:
                teams_by_country[country] = []
            teams_by_country[country].append(team_name)
        
        return teams_by_country


@lru_cache(maxsize=1)
def get_all_teams() -> List[str]:
    """Get all team names"""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT name FROM clubs WHERE name IS NOT NULL AND name != '' ORDER BY name")
        rows = cursor.fetchall()
        return [row[0] for row in rows]


@lru_cache(maxsize=1)
def get_all_countries() -> List[str]:
    """Get all countries with teams"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT DISTINCT comp.country_name 
               FROM clubs c
               LEFT JOIN competitions comp ON c.domestic_competition_id = comp.competition_id
               WHERE comp.country_name IS NOT NULL
               ORDER BY comp.country_name"""
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows if row[0]]


@lru_cache(maxsize=1)
def get_all_referees() -> List[str]:
    """Get all referee names"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT referee FROM games WHERE referee IS NOT NULL AND referee != '' ORDER BY referee"
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]


@lru_cache(maxsize=1)
def get_all_venues() -> List[str]:
    """Get all venue names"""
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT stadium_name FROM clubs WHERE stadium_name IS NOT NULL AND stadium_name != '' ORDER BY stadium_name"
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows]