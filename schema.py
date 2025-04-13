from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Date, Float, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Competition(Base):
    __tablename__ = 'competitions'
    competition_id = Column(Integer, primary_key=True)
    name = Column(String)
    competition_code = Column(String)
    name = Column(String)
    sub_type = Column(String)
    type = Column(String)
    country_id = Column(Integer)
    country_name = Column(String)
    domestic_league_code = Column(String)
    confederation = Column(String)
    
    games = relationship('Game', back_populates='competition')


class Club(Base):
    __tablename__ = 'clubs'
    club_id = Column(Integer, primary_key=True)
    club_code = Column(String)
    name = Column(String)
    domestic_competition_id = Column(Integer, ForeignKey('competitions.competition_id'))
    total_market_value = Column(String)
    squad_size = Column(Integer)
    average_age = Column(Float)
    foreigners_number = Column(Integer)
    foreigners_percentage = Column(String)
    national_team_players = Column(Integer)
    stadium_name = Column(String)
    stadium_seats = Column(Integer)
    net_transfer_record = Column(String)
    coach_name = Column(String)
    last_season = Column(String)
    filename = Column(String)
    url = Column(String)

    players = relationship('Player', back_populates='current_club')
    games_as_home = relationship('Game', back_populates='home_club', foreign_keys='Game.home_club_id')
    games_as_away = relationship('Game', back_populates='away_club', foreign_keys='Game.away_club_id')


class Player(Base):
    __tablename__ = 'players'
    player_id = Column(Integer, primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    name = Column(String)
    last_season = Column(String)
    current_club_id = Column(Integer, ForeignKey('clubs.club_id'))
    player_code = Column(String)
    country_of_birth = Column(String)
    city_of_birth = Column(String)
    country_of_citizenship = Column(String)
    date_of_birth = Column(Date)
    sub_position = Column(String)
    position = Column(String)
    foot = Column(String)
    height_in_cm = Column(Integer)
    contract_expiration_date = Column(String)
    agent_name = Column(String)
    image_url = Column(String)
    url = Column(String)
    current_club_domestic_competition_id = Column(Integer)
    current_club_name = Column(String)
    market_value_in_eur = Column(Float)
    highest_market_value_in_eur = Column(Float)

    current_club = relationship('Club', back_populates='players')
    appearances = relationship('Appearance', back_populates='player')
    events = relationship('GameEvent', back_populates='player')
    valuations = relationship('PlayerValuation', back_populates='player')


class Game(Base):
    __tablename__ = 'games'
    game_id = Column(Integer, primary_key=True)
    competition_id = Column(Integer, ForeignKey('competitions.competition_id'))
    season = Column(String)
    round = Column(String)
    date = Column(Date)
    home_club_id = Column(Integer, ForeignKey('clubs.club_id'))
    away_club_id = Column(Integer, ForeignKey('clubs.club_id'))
    home_club_goals = Column(Integer)
    away_club_goals = Column(Integer)
    home_club_position = Column(Integer)
    away_club_position = Column(Integer)
    home_club_manager_name = Column(String)
    away_club_manager_name = Column(String)
    stadium = Column(String)
    attendance = Column(Integer)
    referee = Column(String)
    url = Column(String)
    home_club_formation = Column(String)
    away_club_formation = Column(String)
    home_club_name = Column(String)
    away_club_name = Column(String)
    aggregate = Column(String)
    competition_type = Column(String)

    competition = relationship('Competition', back_populates='games')
    home_club = relationship('Club', back_populates='games_as_home', foreign_keys=[home_club_id])
    away_club = relationship('Club', back_populates='games_as_away', foreign_keys=[away_club_id])

    club_games = relationship('ClubGame', back_populates='game')
    appearances = relationship('Appearance', back_populates='game')
    events = relationship('GameEvent', back_populates='game')
    lineups = relationship('GameLineup', back_populates='game')


class ClubGame(Base):
    __tablename__ = 'club_games'
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.game_id'))
    club_id = Column(Integer, ForeignKey('clubs.club_id'))
    own_goals = Column(Integer)
    own_position = Column(Integer)
    own_manager_name = Column(String)
    opponent_id = Column(Integer)
    opponent_goals = Column(Integer)
    opponent_position = Column(Integer)
    opponent_manager_name = Column(String)
    hosting = Column(Boolean)
    is_win = Column(Boolean)

    game = relationship('Game', back_populates='club_games')


class Appearance(Base):
    __tablename__ = 'appearances'
    appearance_id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey('games.game_id'))
    player_id = Column(Integer, ForeignKey('players.player_id'))
    player_club_id = Column(Integer)
    player_current_club_id = Column(Integer)
    date = Column(Date)
    player_name = Column(String)
    competition_id = Column(Integer)
    yellow_cards = Column(Integer)
    red_cards = Column(Integer)
    goals = Column(Integer)
    assists = Column(Integer)
    minutes_played = Column(Integer)

    game = relationship('Game', back_populates='appearances')
    player = relationship('Player', back_populates='appearances')


class GameEvent(Base):
    __tablename__ = 'game_events'
    game_event_id = Column(Integer, primary_key=True)
    date = Column(Date)
    game_id = Column(Integer, ForeignKey('games.game_id'))
    minute = Column(Integer)
    type = Column(String)
    club_id = Column(Integer, ForeignKey('clubs.club_id'))
    player_id = Column(Integer, ForeignKey('players.player_id'))
    description = Column(String)
    player_in_id = Column(Integer)
    player_assist_id = Column(Integer)

    game = relationship('Game', back_populates='events')
    player = relationship('Player', back_populates='events')


class PlayerValuation(Base):
    __tablename__ = 'player_valuations'
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'))
    date = Column(Date)
    market_value_in_eur = Column(Float)
    current_club_id = Column(Integer)
    player_club_domestic_competition_id = Column(Integer)

    player = relationship('Player', back_populates='valuations')


class GameLineup(Base):
    __tablename__ = 'game_lineups'
    game_lineups_id = Column(Integer, primary_key=True)
    date = Column(Date)
    game_id = Column(Integer, ForeignKey('games.game_id'))
    player_id = Column(Integer, ForeignKey('players.player_id'))
    club_id = Column(Integer)
    player_name = Column(String)
    type = Column(String)
    position = Column(String)
    number = Column(Integer)
    team_captain = Column(Boolean)

    game = relationship('Game', back_populates='lineups')
    player = relationship('Player')

class Transfer(Base):
    __tablename__ = 'transfers'
    transfer_id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.player_id'))
    transfer_date = Column(Date)
    transfer_season = Column(String)
    from_club_id = Column(Integer, ForeignKey('clubs.club_id'))
    to_club_id = Column(Integer, ForeignKey('clubs.club_id'))
    from_club_name = Column(String)
    to_club_name = Column(String)
    transfer_fee = Column(String)  # String to allow values like "€10m", "Loan", "Free"
    market_value_in_eur = Column(Float)
    player_name = Column(String)

    player = relationship('Player', backref='transfers')
    from_club = relationship('Club', foreign_keys=[from_club_id], backref='transfers_out')
    to_club = relationship('Club', foreign_keys=[to_club_id], backref='transfers_in')


# Create database
engine = create_engine('sqlite:///football.db')
Base.metadata.create_all(engine)

print("✅ All tables created with all columns and relations.")
