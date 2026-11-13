"""
get_nfl_data.py - download NFL data from nflverse and save it locally.

Setup:
    pip install nflreadpy pandas pyarrow

Usage:
    python get_nfl_data.py                 # 2018 through current season (skips 2020)
    python get_nfl_data.py --start 2021    # from 2021
    python get_nfl_data.py --only pbp      # just refresh one dataset

Re-run weekly during the season to pick up new games.

Design notes:
  * 2020 is skipped by default (COVID season: no fans, odd schedules).
  * Everything is saved to data/*.parquet. The modeling code decides what
    counts as "train" vs "test", so the raw data here is never filtered.
"""

import argparse
import time
from pathlib import Path

import nflreadpy as nfl

DATA_DIR = Path("data")
SKIP_SEASONS = {2020}

# Play-by-play has ~370 columns; these are the ones the project uses.
PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "home_team",
    "away_team", "posteam", "defteam", "qtr", "down", "ydstogo",
    "yardline_100", "goal_to_go", "game_seconds_remaining",
    "score_differential", "posteam_score", "defteam_score", "drive",
    "fixed_drive", "fixed_drive_result", "play_type", "yards_gained",
    "epa", "wp", "wpa", "vegas_wp", "success", "first_down", "penalty",
    "penalty_yards", "pass", "rush", "pass_attempt", "complete_pass",
    "qb_dropback", "qb_scramble", "qb_hit", "sack", "interception",
    "fumble", "fumble_lost", "fumble_forced", "touchdown", "pass_touchdown",
    "rush_touchdown", "return_touchdown", "td_team", "safety",
    "kickoff_attempt", "punt_attempt", "field_goal_attempt",
    "field_goal_result", "two_point_attempt", "third_down_converted",
    "third_down_failed", "fourth_down_converted", "fourth_down_failed",
    "special_teams_play", "return_yards", "air_yards", "yards_after_catch",
    "pass_location", "pass_length", "run_location", "run_gap", "shotgun",
    "no_huddle", "xpass", "pass_oe", "cpoe", "xyac_epa",
    "passer_player_id", "passer_player_name", "receiver_player_id",
    "receiver_player_name", "rusher_player_id", "rusher_player_name",
    "kick_returner_player_name", "punt_returner_player_name",
    "interception_player_id", "td_player_id",
]

# Formation / personnel / coverage info (available 2016-2025 only).
PART_COLS = [
    "nflverse_game_id", "play_id", "possession_team", "offense_formation",
    "offense_personnel", "defenders_in_box", "defense_personnel",
    "number_of_pass_rushers", "was_pressure", "time_to_throw",
    "defense_man_zone_type", "defense_coverage_type", "route", "ngs_air_yards",
]


def keep_cols(df, cols):
    return df[[c for c in cols if c in df.columns]]


def seasons_for(loader_name, seasons, current):
    """Some datasets don't cover every season."""
    if loader_name == "participation":
        return [s for s in seasons if s <= 2025]  # not published for 2026 yet
    return seasons


def with_retry(fn, seasons, tries=4):
    """Data comes from GitHub, which occasionally times out."""
    for attempt in range(1, tries + 1):
        try:
            return fn(seasons)
        except Exception as e:
            if attempt == tries:
                raise
            print(f"    retry {attempt} after error: {str(e)[:80]}")
            time.sleep(3 * attempt)


def download(name, loader, seasons, current, trim=None, per_season=False):
    """per_season=True downloads one year at a time (for the big datasets)."""
    seasons = seasons_for(name, seasons, current)
    print(f"{name}  ({seasons[0]}-{seasons[-1]})")
    if per_season:
        import pandas as pd
        parts = []
        for s in seasons:
            d = with_retry(loader, [s]).to_pandas()
            parts.append(trim(d) if trim else d)
            print(f"    {s}: {len(d):,} rows")
        df = pd.concat(parts, ignore_index=True)
    else:
        df = with_retry(loader, seasons).to_pandas()  # Polars -> pandas
        df = trim(df) if trim else df
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  saved {len(df):,} rows x {df.shape[1]} cols -> {path}\n")


def main():
    current = nfl.get_current_season()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=current)
    ap.add_argument("--only", help="download just one dataset by name")
    args = ap.parse_args()
    seasons = [s for s in range(args.start, args.end + 1) if s not in SKIP_SEASONS]

    DATA_DIR.mkdir(exist_ok=True)
    print(f"Seasons: {seasons}\n")

    # name: (loader, trim function, per_season)
    datasets = {
        "schedules":     (nfl.load_schedules, None, False),   # games, scores, spreads, weather
        "team_stats":    (nfl.load_team_stats, None, False),  # team-game box score stats
        "player_stats":  (nfl.load_player_stats, None, False),  # player-game stats
        "pbp":           (nfl.load_pbp, lambda d: keep_cols(d, PBP_COLS), True),
        "participation": (nfl.load_participation, lambda d: keep_cols(d, PART_COLS), True),
        "injuries":      (nfl.load_injuries, None, False),    # weekly injury reports
        "snap_counts":   (nfl.load_snap_counts, None, False),
    }

    for name, (loader, trim, per_season) in datasets.items():
        if args.only and name != args.only:
            continue
        download(name, loader, seasons, current, trim, per_season)

    print("Done. Load later with: pd.read_parquet('data/pbp.parquet')")


if __name__ == "__main__":
    main()
