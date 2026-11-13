"""
profiles.py - play-style profiles for every team (offense and defense), pre-game.
  core     : from play-by-play only (available for every season incl. 2026)
  extended : adds personnel / man-zone / blitz / pressure (participation data, 2018-2025 only)
Pre-game profile = recency-weighted average of the team's previous games, carrying last season at half weight.
"""
import re
import numpy as np, pandas as pd

DATA = "data"
OFF_CORE = ["pass_rate_neutral", "pass_oe", "shotgun_rate", "no_huddle_rate", "plays_pg", "epa_pass", "epa_rush",
            "explosive_rate", "deep_rate", "sack_rate_allowed"]
DEF_CORE = ["epa_pass_allowed", "epa_rush_allowed", "explosive_allowed", "deep_rate_faced", "sack_rate_gen", "plays_faced"]
OFF_EXT = ["pers_11", "pers_12", "pers_heavy", "empty_rate"]
DEF_EXT = ["man_rate", "blitz_rate", "pressure_rate", "two_high_rate"]


def personnel_code(s):
    if not isinstance(s, str): return None
    rb = re.search(r"(\d) RB", s); te = re.search(r"(\d) TE", s)
    return f"{rb.group(1) if rb else 0}{te.group(1) if te else 0}"


def game_stats():
    pbp = pd.read_parquet(f"{DATA}/pbp.parquet")
    pbp = pbp[(pbp.season_type == "REG") & pbp.posteam.notna() & pbp.defteam.notna()]
    scrim = pbp[pbp.play_type.isin(["pass", "run"]) | (pbp.qb_dropback == 1)].copy()
    scrim["is_db"] = (scrim.qb_dropback == 1).astype(int)
    scrim["is_run"] = ((scrim.play_type == "run") & (scrim.qb_scramble == 0)).astype(int)
    scrim["explosive"] = (scrim.yards_gained >= 20).astype(int)
    scrim["deep"] = ((scrim.pass_attempt == 1) & (scrim.air_yards >= 20)).astype(int)
    scrim["att"] = scrim.pass_attempt.fillna(0)
    neutral = scrim[(scrim.wp.between(0.2, 0.8)) & scrim.down.isin([1, 2]) & (scrim.qtr <= 3) & (scrim.is_db + scrim.is_run > 0)].copy()
    neutral["pass_over_exp"] = (neutral.is_db - neutral.xpass) * 100

    def agg(side):
        g = scrim.groupby(["game_id", side])
        out = pd.DataFrame({
            "plays": g.size(),
            "shotgun": g.shotgun.mean(), "nohuddle": g.no_huddle.mean(),
            "epa_pass": scrim[scrim.is_db == 1].groupby(["game_id", side]).epa.mean(),
            "epa_rush": scrim[scrim.is_run == 1].groupby(["game_id", side]).epa.mean(),
            "explosive": g.explosive.mean(),
            "deep": g.deep.sum() / g.att.sum().replace(0, np.nan),
            "sack_rate": scrim[scrim.is_db == 1].groupby(["game_id", side]).sack.mean(),
        })
        n = neutral.groupby(["game_id", side])
        out["pass_rate_neutral"] = n.is_db.mean(); out["pass_oe"] = n.pass_over_exp.mean()
        return out.reset_index().rename(columns={side: "team"})

    off = agg("posteam").rename(columns={"plays": "plays_pg", "shotgun": "shotgun_rate", "nohuddle": "no_huddle_rate",
                                         "explosive": "explosive_rate", "deep": "deep_rate", "sack_rate": "sack_rate_allowed"})
    dfn = agg("defteam").rename(columns={"plays": "plays_faced", "epa_pass": "epa_pass_allowed", "epa_rush": "epa_rush_allowed",
                                         "explosive": "explosive_allowed", "deep": "deep_rate_faced", "sack_rate": "sack_rate_gen"})
    off, dfn = off[["game_id", "team"] + OFF_CORE], dfn[["game_id", "team"] + DEF_CORE]

    # ---- extended (participation) ----
    part = pd.read_parquet(f"{DATA}/participation.parquet")
    m = scrim.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")
    m["pc"] = m.offense_personnel.map(personnel_code)
    m["p11"] = (m.pc == "11").astype(float); m["p12"] = (m.pc == "12").astype(float)
    m["heavy"] = m.pc.isin(["21", "22", "20", "13", "23", "02", "01"]).astype(float)
    m["empty"] = (m.offense_formation == "EMPTY").astype(float)
    eo = m.groupby(["game_id", "posteam"]).agg(pers_11=("p11", "mean"), pers_12=("p12", "mean"),
                                               pers_heavy=("heavy", "mean"), empty_rate=("empty", "mean")).reset_index().rename(columns={"posteam": "team"})
    d = m[m.is_db == 1].copy()
    d["man"] = np.where(d.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"]), (d.defense_man_zone_type == "MAN_COVERAGE").astype(float), np.nan)
    d["blitz"] = np.where(d.number_of_pass_rushers.notna(), (d.number_of_pass_rushers >= 5).astype(float), np.nan)
    d["press"] = d.was_pressure.map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}).astype(float)
    d["two_high"] = np.where(d.defense_coverage_type.notna(), d.defense_coverage_type.isin(["COVER_2", "COVER_4", "COVER_6", "2_MAN"]).astype(float), np.nan)
    ed = d.groupby(["game_id", "defteam"]).agg(man_rate=("man", "mean"), blitz_rate=("blitz", "mean"),
                                               pressure_rate=("press", "mean"), two_high_rate=("two_high", "mean")).reset_index().rename(columns={"defteam": "team"})
    return off.merge(eo, on=["game_id", "team"], how="left"), dfn.merge(ed, on=["game_id", "team"], how="left")


def pregame_profiles(stats, cols, half=0.5, decay=0.9, prior_w=1.5):
    """For each (game_id, team): profile from games BEFORE it. Recency-weighted, last season carried at half weight."""
    sch = pd.read_parquet(f"{DATA}/schedules.parquet")[["game_id", "season", "week", "gameday"]]
    s = stats.merge(sch, on="game_id").sort_values(["team", "season", "week"])
    mu = s[s.season <= 2021][cols].mean()
    rows = []
    for team, tdf in s.groupby("team"):
        S = pd.Series(0.0, index=cols); W = pd.Series(0.0, index=cols); prev = None
        for _, r in tdf.iterrows():
            if prev is not None and r.season != prev:
                S, W = S * half, W * half
            prev = r.season
            prof = (S + prior_w * mu) / (W + prior_w)
            rows.append({"game_id": r.game_id, "team": team, **prof.to_dict()})
            x = r[cols].astype(float); ok = x.notna().astype(float)
            S = S * decay + x.fillna(0) * ok; W = W * decay + ok
        # (future/unplayed games would need a final profile; handled in the app step)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    off, dfn = game_stats()
    off.to_parquet(f"{DATA}/off_game_stats.parquet", index=False); dfn.to_parquet(f"{DATA}/def_game_stats.parquet", index=False)
    po = pregame_profiles(off, OFF_CORE + OFF_EXT); pdn = pregame_profiles(dfn, DEF_CORE + DEF_EXT)
    po.to_parquet(f"{DATA}/off_profiles.parquet", index=False); pdn.to_parquet(f"{DATA}/def_profiles.parquet", index=False)
    print(off.shape, dfn.shape, po.shape, pdn.shape)
    print(po.describe().T[["mean", "std"]].round(3))
    print(pdn.describe().T[["mean", "std"]].round(3))
