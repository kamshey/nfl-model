"""
players.py - player-game dataset with only pre-game information:
  * recent usage/production (recency-weighted), shares of team volume
  * game context: Vegas-implied team total, home/away, team pass tendency
  * opponent: pass-defense profile, yards allowed to the player's position
  * man-vs-zone matchup: how the receiver performs vs man / vs zone coverage, and how the defense plays + fares against each
"""
import numpy as np, pandas as pd
from profiles import pregame_profiles

DATA = "data"
STATS = ["targets", "receptions", "receiving_yards", "receiving_tds", "carries", "rushing_yards", "rushing_tds",
         "attempts", "passing_yards", "passing_tds"]


def decayed_pre(df, key, vals, decay):
    """Per entity (ordered rows): decayed sums of `vals` over PREVIOUS rows only."""
    out = np.zeros((len(df), len(vals)))
    X = df[vals].fillna(0).values
    pos = {k: np.where(df[key].values == k)[0] for k in pd.unique(df[key].values)} if isinstance(key, str) else None
    for k, idx in pos.items():
        S = np.zeros(len(vals))
        for i in idx:
            out[i] = S
            S = S * decay + X[i]
    return pd.DataFrame(out, index=df.index, columns=[f"pre_{v}" for v in vals])


def build(verbose=True):
    sch = pd.read_parquet(f"{DATA}/schedules.parquet")
    sch = sch[sch.game_type == "REG"][["game_id", "season", "week", "home_team", "away_team", "spread_line", "total_line", "gameday"]]
    ps = pd.read_parquet(f"{DATA}/player_stats.parquet")
    ps = ps[(ps.season_type == "REG") & ps.position.isin(["QB", "RB", "WR", "TE"])].copy()
    ps = ps.drop(columns=["season", "week"]).merge(sch, on="game_id")
    ps = ps.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    for c in STATS: ps[c] = ps[c].fillna(0)

    # ---- shares of team volume in the game, then recency-weighted history (shifted: pre-game only) ----
    tt = ps.groupby(["game_id", "team"])[["targets", "carries", "attempts"]].transform("sum")
    ps["tgt_share"] = ps.targets / tt.targets.replace(0, np.nan)
    ps["car_share"] = ps.carries / tt.carries.replace(0, np.nan)
    hist_cols = STATS + ["tgt_share", "car_share"]
    g = ps.groupby("player_id")
    for c in hist_cols:
        ps[f"e_{c}"] = g[c].transform(lambda s: s.shift(1).ewm(alpha=0.3, min_periods=1).mean())
    ps["n_prior"] = g.cumcount()
    ps["games_gap"] = g.season.transform(lambda s: s.diff()).fillna(0)

    # ---- game context ----
    home = ps.team == ps.home_team
    ps["is_home"] = home.astype(int)
    ps["team_total"] = np.where(home, (ps.total_line + ps.spread_line) / 2, (ps.total_line - ps.spread_line) / 2)
    ps["spread_for"] = np.where(home, ps.spread_line, -ps.spread_line)
    po = pd.read_parquet(f"{DATA}/off_profiles.parquet")[["game_id", "team", "pass_rate_neutral", "plays_pg", "pass_oe"]]
    pdn = pd.read_parquet(f"{DATA}/def_profiles.parquet")[["game_id", "team", "epa_pass_allowed", "epa_rush_allowed", "sack_rate_gen",
                                                            "man_rate", "blitz_rate", "pressure_rate", "two_high_rate"]]
    ps = ps.merge(po.rename(columns={"pass_rate_neutral": "tm_pass_rate", "plays_pg": "tm_plays", "pass_oe": "tm_pass_oe"}), on=["game_id", "team"], how="left")
    ps = ps.merge(pdn.rename(columns={"team": "opponent_team"}), on=["game_id", "opponent_team"], how="left")

    # ---- yards the opponent has been allowing to this position (recency-weighted, pre-game) ----
    pos_allowed = ps.groupby(["game_id", "opponent_team", "position"]).agg(rec=("receiving_yards", "sum"), rush=("rushing_yards", "sum"), pas=("passing_yards", "sum")).reset_index()
    wide = pos_allowed.pivot_table(index=["game_id", "opponent_team"], columns="position", values=["rec", "rush", "pas"], fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide[["rec_WR", "rec_TE", "rec_RB", "rush_RB", "pas_QB"]].reset_index().rename(columns={"opponent_team": "team"})
    pa = pregame_profiles(wide, ["rec_WR", "rec_TE", "rec_RB", "rush_RB", "pas_QB"]).rename(columns=lambda c: c if c in ("game_id", "team") else "allow_" + c)
    ps = ps.merge(pa.rename(columns={"team": "opponent_team"}), on=["game_id", "opponent_team"], how="left")

    # ---- man vs zone ----
    pbp = pd.read_parquet(f"{DATA}/pbp.parquet", columns=["game_id", "play_id", "season_type", "pass_attempt", "receiver_player_id", "epa", "defteam", "sack"])
    pbp = pbp[(pbp.season_type == "REG") & (pbp.pass_attempt == 1) & pbp.receiver_player_id.notna() & pbp.epa.notna()]
    part = pd.read_parquet(f"{DATA}/participation.parquet", columns=["nflverse_game_id", "play_id", "defense_man_zone_type"])
    t = pbp.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="inner")
    t = t[t.defense_man_zone_type.isin(["MAN_COVERAGE", "ZONE_COVERAGE"])]
    t["man"] = (t.defense_man_zone_type == "MAN_COVERAGE").astype(float)
    lg = {"man": t[t.man == 1].epa.mean(), "zone": t[t.man == 0].epa.mean()}
    t["n_man"], t["n_zone"] = t.man, 1 - t.man
    t["e_man"], t["e_zone"] = t.epa * t.man, t.epa * (1 - t.man)
    rc = t.groupby(["game_id", "receiver_player_id"])[["n_man", "n_zone", "e_man", "e_zone"]].sum().reset_index().rename(columns={"receiver_player_id": "player_id"})
    dc = t.groupby(["game_id", "defteam"])[["n_man", "n_zone", "e_man", "e_zone"]].sum().reset_index().rename(columns={"defteam": "opponent_team"})
    mzc = ["n_man", "n_zone", "e_man", "e_zone"]
    ps = ps.merge(rc, on=["game_id", "player_id"], how="left")
    for c in mzc: ps[c] = ps[c].fillna(0)
    pre = decayed_pre(ps, "player_id", mzc, decay=0.97)
    K = 20
    ps["pl_man_pt"] = (pre.pre_e_man + K * lg["man"]) / (pre.pre_n_man + K)
    ps["pl_zone_pt"] = (pre.pre_e_zone + K * lg["zone"]) / (pre.pre_n_zone + K)
    ps["man_edge"] = ps.pl_man_pt - ps.pl_zone_pt                      # + = better vs man than vs zone
    ps["pl_mz_n"] = pre.pre_n_man + pre.pre_n_zone
    # defense side: how it fares vs receivers when in man / in zone
    dd = ps[["game_id", "opponent_team", "season", "week"]].drop_duplicates().merge(dc, on=["game_id", "opponent_team"], how="left").fillna(0)
    dd = dd.sort_values(["opponent_team", "season", "week"]).reset_index(drop=True)
    dpre = decayed_pre(dd, "opponent_team", mzc, decay=0.9)
    dd["def_man_pt"] = (dpre.pre_e_man + 40 * lg["man"]) / (dpre.pre_n_man + 40)
    dd["def_zone_pt"] = (dpre.pre_e_zone + 40 * lg["zone"]) / (dpre.pre_n_zone + 40)
    ps = ps.merge(dd[["game_id", "opponent_team", "def_man_pt", "def_zone_pt"]], on=["game_id", "opponent_team"], how="left")
    mr = ps.man_rate.fillna(0.357)
    ps["mz_fit"] = ps.man_edge * (mr - 0.357)                            # receiver's man-vs-zone skill x how man-heavy this defense is
    ps["def_mix_weak"] = mr * (ps.def_man_pt - lg["man"]) + (1 - mr) * (ps.def_zone_pt - lg["zone"])   # + = defense is leaky given its mix
    ps["exp_epa_pt"] = mr * ps.pl_man_pt + (1 - mr) * ps.pl_zone_pt      # receiver's expected efficiency vs this defense's coverage mix
    return ps


if __name__ == "__main__":
    ps = build()
    ps.to_parquet(f"{DATA}/player_dataset.parquet", index=False)
    print(ps.shape); print(ps[["season", "week"]].agg(["min", "max"]).T)
    r = ps[(ps.position == "WR") & (ps.season == 2024) & (ps.n_prior > 20)].head(3)
    print(r[["player_display_name", "week", "opponent_team", "e_targets", "e_receiving_yards", "team_total", "man_edge", "man_rate", "mz_fit", "def_mix_weak"]].round(3).to_string())
    print(ps[["man_edge", "mz_fit", "def_mix_weak", "man_rate"]].describe().T[["mean", "std", "min", "max"]].round(3))
