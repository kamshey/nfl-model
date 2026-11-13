"""
ratings_v2.py - rating engine with (1) luck adjustment, (2) QB-change feature, (3) injuries.
Everything is pre-game information only. Flags let us ablate each piece.
"""
import re
import numpy as np, pandas as pd
from ratings import Ratings, METRICS, build_games, DATA

LUCK_W = {"fum": 3.5, "int": 3.0, "td": 5.0, "fg": 2.0}   # points per event (rounded from the 2018-21 post-mortem)


def clean_epa(pbp):
    """EPA with the noisiest pieces removed: fumble plays dropped (recovery is a coin flip), extreme plays capped."""
    p = pbp[pbp.posteam.notna() & pbp.epa.notna()].copy()
    p = p[p.fumble != 1]
    p["epa"] = p.epa.clip(-4, 4)
    return p


def team_game_table(clean):
    pbp = pd.read_parquet(f"{DATA}/pbp.parquet")
    pbp = clean_epa(pbp) if clean else pbp[pbp.posteam.notna() & pbp.epa.notna()]
    db, ru = pbp[pbp.qb_dropback == 1], pbp[(pbp.play_type == "run") & (pbp.qb_scramble == 0)]
    a = db.groupby(["game_id", "posteam"]).agg(pass_epa=("epa", "mean"), n_db=("epa", "size"))
    b = ru.groupby(["game_id", "posteam"]).agg(rush_epa=("epa", "mean"), n_ru=("epa", "size"))
    q = (db.dropna(subset=["passer_player_id"]).groupby(["game_id", "posteam", "passer_player_id"])
           .agg(q_epa=("epa", "sum"), q_n=("epa", "size")).reset_index())
    return a.join(b, how="outer").reset_index(), q


def luck_points():
    """Net luck, in points, per (game_id, team): >0 means the bounces favored that team."""
    pbp = pd.read_parquet(f"{DATA}/pbp.parquet")
    sch = pd.read_parquet(f"{DATA}/schedules.parquet")
    scr = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.posteam.notna()]
    g = scr.groupby(["game_id", "posteam"])
    tg = pd.DataFrame({"fumbles": g.fumble.sum(), "fumbles_lost": g.fumble_lost.sum(),
                       "ints": g.interception.sum(), "attempts": g.pass_attempt.sum()}).fillna(0).reset_index().rename(columns={"posteam": "team"})
    rt = pbp[(pbp.return_touchdown == 1) & pbp.td_team.notna()].groupby(["game_id", "td_team"]).size().rename("ret_td").reset_index().rename(columns={"td_team": "team"})
    tg = tg.merge(rt, on=["game_id", "team"], how="left").fillna({"ret_td": 0})
    fg = pbp[(pbp.field_goal_attempt == 1) & pbp.posteam.notna()].copy()
    fg["p_make"] = np.interp(fg.yardline_100 + 17, [20, 30, 40, 45, 50, 55, 60], [0.99, 0.95, 0.87, 0.82, 0.74, 0.65, 0.50])
    fg["miss"] = (fg.field_goal_result != "made").astype(int)
    fgg = fg.groupby(["game_id", "posteam"]).agg(fg_miss=("miss", "sum"), fg_exp=("p_make", lambda p: (1 - p).sum())).reset_index().rename(columns={"posteam": "team"})
    tg = tg.merge(fgg, on=["game_id", "team"], how="left").fillna(0)
    tg["exp_ints"] = tg.ints.sum() / tg.attempts.sum() * tg.attempts      # league INT rate (minor, non-predictive use)
    st = pd.concat([sch[["game_id", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"}),
                    sch[["game_id", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})])
    T = st.merge(tg, on=["game_id", "team"], how="left")
    O = tg.add_prefix("o_").rename(columns={"o_game_id": "game_id", "o_team": "opp"})
    T = T.merge(O, on=["game_id", "opp"], how="left").fillna(0)
    fum = (0.5 * T.fumbles - T.fumbles_lost) + (T.o_fumbles_lost - 0.5 * T.o_fumbles)
    ints = (T.exp_ints - T.ints) + (T.o_ints - T.o_exp_ints)
    td = T.ret_td - T.o_ret_td
    fgl = (T.o_fg_miss - T.o_fg_exp) + (T.fg_exp - T.fg_miss)
    T["luck_pts"] = LUCK_W["fum"] * fum + LUCK_W["int"] * ints + LUCK_W["td"] * td + LUCK_W["fg"] * fgl
    return T.set_index(["game_id", "team"]).luck_pts


def norm_name(s):
    s = re.sub(r"[.'`]", "", str(s).lower()).strip()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)


def injury_features():
    """Snap-weighted 'who's missing' per (season, week, team): offense and defense, QBs excluded."""
    inj = pd.read_parquet(f"{DATA}/injuries.parquet")
    snp = pd.read_parquet(f"{DATA}/snap_counts.parquet")
    inj = inj[inj.game_type == "REG"].copy()
    inj["w"] = inj.report_status.map({"Out": 1.0, "Doubtful": 0.8, "Questionable": 0.2}).fillna(0)
    inj = inj[(inj.w > 0) & (inj.position != "QB")]
    inj["nm"] = inj.full_name.map(norm_name)
    snp = snp[snp.game_type == "REG"].copy()
    snp["nm"] = snp.player.map(norm_name)
    snp["tkey"] = (snp.season * 100 + snp.week).astype("int64")
    snp = snp.sort_values(["nm", "tkey"])
    for c in ["offense_pct", "defense_pct"]:
        snp[c + "_r"] = snp.groupby("nm")[c].transform(lambda s: s.rolling(4, min_periods=1).mean())
    snp = snp.drop_duplicates(["nm", "tkey"], keep="last")
    inj["tkey"] = (inj.season * 100 + inj.week).astype("int64")
    inj = inj.sort_values("tkey")
    m = pd.merge_asof(inj, snp[["nm", "tkey", "offense_pct_r", "defense_pct_r"]].sort_values("tkey"),
                      on="tkey", by="nm", allow_exact_matches=False)
    m[["offense_pct_r", "defense_pct_r"]] = m[["offense_pct_r", "defense_pct_r"]].fillna(0)
    m["inj_off"], m["inj_def"] = m.w * m.offense_pct_r, m.w * m.defense_pct_r
    return m.groupby(["season", "week", "team"])[["inj_off", "inj_def"]].sum().reset_index()


def build_features(alpha=0.06, shrink=0.8, luck=True, qb_delta=True, injuries=True):
    games = build_games()
    tg, qbg = team_game_table(clean=luck)
    tg_idx = tg.set_index(["game_id", "posteam"])
    qb_by_game = {k: v for k, v in qbg.groupby(["game_id", "posteam"])}
    L = luck_points() if luck else None
    R = Ratings(alpha=alpha, shrink=shrink)
    team_qb = {}                      # recent average QB rating per team (to measure "QB change")
    rows, prev = [], None
    for (season, week), wk in games.groupby(["season", "week"], sort=True):
        if prev is not None and season != prev:
            R.new_season(years_gap=season - prev)
        prev = season
        for _, g in wk.iterrows():
            h, a = g.home_team, g.away_team
            f = {"game_id": g.game_id, "season": season, "week": week, "game_type": g.game_type, "home_team": h, "away_team": a}
            for m in METRICS:
                f[f"h_off_{m}"], f[f"h_def_{m}"] = R.g(R.off[m], h), R.g(R.dfn[m], h)
                f[f"a_off_{m}"], f[f"a_def_{m}"] = R.g(R.off[m], a), R.g(R.dfn[m], a)
                f[f"h_exp_{m}"] = R.mu[m] + f[f"h_off_{m}"] + f[f"a_def_{m}"]
                f[f"a_exp_{m}"] = R.mu[m] + f[f"a_off_{m}"] + f[f"h_def_{m}"]
            f["h_qb"], f["h_qb_n"] = R.qb_rating(g.home_qb_id)
            f["a_qb"], f["a_qb_n"] = R.qb_rating(g.away_qb_id)
            f["h_qb_delta"] = f["h_qb"] - team_qb.get(h, f["h_qb"])      # today's QB vs the team's recent QB
            f["a_qb_delta"] = f["a_qb"] - team_qb.get(a, f["a_qb"])
            f["_hq"], f["_aq"] = f["h_qb"], f["a_qb"]
            f["rest_diff"] = g.home_rest - g.away_rest
            f["div_game"], f["neutral"] = int(g.div_game), int(g.location == "Neutral")
            dome = g.roof in ("dome", "closed")
            f["dome"] = int(dome)
            f["wind"] = 0.0 if dome or pd.isna(g.wind) else g.wind
            f["temp"] = 70.0 if dome or pd.isna(g.temp) else g.temp
            f["spread_line"], f["total_line"] = g.spread_line, g.total_line
            f["home_ml"], f["away_ml"] = g.home_moneyline, g.away_moneyline
            f["home_score"], f["away_score"] = g.home_score, g.away_score
            rows.append(f)
        for _, g in wk.iterrows():
            if pd.isna(g.home_score):
                continue
            h, a = g.home_team, g.away_team
            for team, opp, pts, qid in [(h, a, g.home_score, g.home_qb_id), (a, h, g.away_score, g.away_qb_id)]:
                adj = pts - (L.get((g.game_id, team), 0.0) / 2 if luck else 0.0)   # strip half of net luck from offense...
                R.update_team("pts", team, opp, adj)                                # ...the opponent's offense gets the other half
                key = (g.game_id, team)
                if key in tg_idx.index:
                    r = tg_idx.loc[key]
                    if pd.notna(r.pass_epa) and r.n_db >= 10: R.update_team("pass", team, opp, r.pass_epa)
                    if pd.notna(r.rush_epa) and r.n_ru >= 5: R.update_team("rush", team, opp, r.rush_epa)
                if key in qb_by_game:
                    for _, q in qb_by_game[key].iterrows(): R.update_qb(q.passer_player_id, q.q_epa, q.q_n)
                r_used = R.qb_rating(qid)[0]
                team_qb[team] = r_used if team not in team_qb else 0.65 * team_qb[team] + 0.35 * r_used
    F = pd.DataFrame(rows).drop(columns=["_hq", "_aq"])
    F["margin"], F["total"] = F.home_score - F.away_score, F.home_score + F.away_score
    if injuries:
        I = injury_features()
        for side, tcol in [("h", "home_team"), ("a", "away_team")]:
            F = F.merge(I.rename(columns={"team": tcol, "inj_off": f"{side}_inj_off", "inj_def": f"{side}_inj_def"}),
                        on=["season", "week", tcol], how="left")
        F[["h_inj_off", "h_inj_def", "a_inj_off", "a_inj_def"]] = F[["h_inj_off", "h_inj_def", "a_inj_off", "a_inj_def"]].fillna(0)
        F["inj_off_diff"], F["inj_def_diff"] = F.h_inj_off - F.a_inj_off, F.h_inj_def - F.a_inj_def
    return F
