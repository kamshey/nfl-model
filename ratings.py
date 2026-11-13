"""
ratings.py - build pre-game features for every game, using ONLY information
available before kickoff.

Core idea: an opponent-adjusted rating system (like Elo, but for several
metrics). For each team we track an offense and defense rating on:
    pass  - EPA per dropback
    rush  - EPA per designed run
    pts   - points scored / allowed
After every week, each team-game result is compared to what the ratings
expected (league average + own offense + opponent defense). The surprise
nudges both teams' ratings. Quarterbacks get their own rating too, since a QB
change is the biggest roster swing in football.

Leakage guard: all games in a week get features BEFORE any of that week's
results are used to update ratings.
"""

import numpy as np
import pandas as pd

DATA = "data"
METRICS = ["pass", "rush", "pts"]
INIT_MU = {"pass": 0.0, "rush": -0.05, "pts": 22.0}


def team_game_table():
    """One row per (game, offense): passing/rushing EPA and counts, from pbp."""
    pbp = pd.read_parquet(f"{DATA}/pbp.parquet")
    pbp = pbp[pbp.posteam.notna() & pbp.epa.notna()]
    db = pbp[pbp.qb_dropback == 1]
    ru = pbp[(pbp.play_type == "run") & (pbp.qb_scramble == 0)]

    a = db.groupby(["game_id", "posteam"]).agg(pass_epa=("epa", "mean"), n_db=("epa", "size"))
    b = ru.groupby(["game_id", "posteam"]).agg(rush_epa=("epa", "mean"), n_ru=("epa", "size"))
    tg = a.join(b, how="outer").reset_index()

    # QB-level dropback EPA per game (the passer with the most dropbacks)
    q = (db.dropna(subset=["passer_player_id"])
           .groupby(["game_id", "posteam", "passer_player_id"])
           .agg(q_epa=("epa", "sum"), q_n=("epa", "size")).reset_index())
    return tg, q


def build_games():
    s = pd.read_parquet(f"{DATA}/schedules.parquet")
    s = s.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    return s


class Ratings:
    def __init__(self, alpha=0.12, shrink=0.6, qb_k=150, qb_decay=0.92, mu_rate=0.003):
        self.alpha, self.shrink = alpha, shrink
        self.qb_k, self.qb_decay, self.mu_rate = qb_k, qb_decay, mu_rate
        self.mu = dict(INIT_MU)
        self.off = {m: {} for m in METRICS}
        self.dfn = {m: {} for m in METRICS}
        self.qb = {}  # id -> [decayed epa sum, decayed dropbacks]
        self.games_seen = {}

    def g(self, d, t):
        return d.get(t, 0.0)

    def new_season(self, years_gap=1):
        k = self.shrink ** years_gap  # bigger gap -> more regression to the mean
        for m in METRICS:
            for t in self.off[m]:
                self.off[m][t] *= k
            for t in self.dfn[m]:
                self.dfn[m][t] *= k
        for qid in self.qb:
            self.qb[qid][0] *= 0.7
            self.qb[qid][1] *= 0.7

    def qb_rating(self, qid):
        if qid is None or qid != qid or qid not in self.qb:
            return self.mu["pass"] - 0.10, 0.0  # unknown/rookie QB: below average prior
        e, n = self.qb[qid]
        prior = self.mu["pass"] - 0.05
        return (e + self.qb_k * prior) / (n + self.qb_k), n

    def update_team(self, m, team, opp, obs):
        exp = self.mu[m] + self.g(self.off[m], team) + self.g(self.dfn[m], opp)
        r = obs - exp
        self.off[m][team] = self.g(self.off[m], team) + self.alpha * r
        self.dfn[m][opp] = self.g(self.dfn[m], opp) + self.alpha * r
        self.mu[m] += self.mu_rate * (obs - self.mu[m])

    def update_qb(self, qid, epa_sum, n):
        e, nn = self.qb.get(qid, [0.0, 0.0])
        self.qb[qid] = [e * self.qb_decay + epa_sum, nn * self.qb_decay + n]


def build_features(alpha=0.12, shrink=0.6, verbose=False):
    games = build_games()
    tg, qb_games = team_game_table()
    tg_idx = tg.set_index(["game_id", "posteam"])
    qb_by_game = {k: v for k, v in qb_games.groupby(["game_id", "posteam"])}

    R = Ratings(alpha=alpha, shrink=shrink)
    rows = []
    prev_season = None
    for (season, week), wk in games.groupby(["season", "week"], sort=True):
        if prev_season is not None and season != prev_season:
            R.new_season(years_gap=season - prev_season)  # 2019->2021 counts as 2
        prev_season = season

        # 1) features for every game this week (pre-game info only)
        for _, g in wk.iterrows():
            h, a = g.home_team, g.away_team
            f = {"game_id": g.game_id, "season": season, "week": week, "game_type": g.game_type,
                 "home_team": h, "away_team": a}
            for m in METRICS:
                f[f"h_off_{m}"] = R.g(R.off[m], h)
                f[f"h_def_{m}"] = R.g(R.dfn[m], h)
                f[f"a_off_{m}"] = R.g(R.off[m], a)
                f[f"a_def_{m}"] = R.g(R.dfn[m], a)
                f[f"h_exp_{m}"] = R.mu[m] + f[f"h_off_{m}"] + f[f"a_def_{m}"]
                f[f"a_exp_{m}"] = R.mu[m] + f[f"a_off_{m}"] + f[f"h_def_{m}"]
            f["h_qb"], f["h_qb_n"] = R.qb_rating(g.home_qb_id)
            f["a_qb"], f["a_qb_n"] = R.qb_rating(g.away_qb_id)
            f["rest_diff"] = g.home_rest - g.away_rest
            f["div_game"] = int(g.div_game)
            f["neutral"] = int(g.location == "Neutral")
            dome = g.roof in ("dome", "closed")
            f["dome"] = int(dome)
            f["wind"] = 0.0 if dome or pd.isna(g.wind) else g.wind
            f["temp"] = 70.0 if dome or pd.isna(g.temp) else g.temp
            f["spread_line"] = g.spread_line      # market: home margin expected (+ = home favored)
            f["total_line"] = g.total_line
            f["home_ml"], f["away_ml"] = g.home_moneyline, g.away_moneyline
            f["home_score"], f["away_score"] = g.home_score, g.away_score
            rows.append(f)

        # 2) now learn from this week's results
        for _, g in wk.iterrows():
            if pd.isna(g.home_score):
                continue
            h, a = g.home_team, g.away_team
            for team, opp, pts in [(h, a, g.home_score), (a, h, g.away_score)]:
                R.update_team("pts", team, opp, pts)
                key = (g.game_id, team)
                if key in tg_idx.index:
                    r = tg_idx.loc[key]
                    if pd.notna(r.pass_epa) and r.n_db >= 10:
                        R.update_team("pass", team, opp, r.pass_epa)
                    if pd.notna(r.rush_epa) and r.n_ru >= 5:
                        R.update_team("rush", team, opp, r.rush_epa)
                if key in qb_by_game:
                    for _, q in qb_by_game[key].iterrows():
                        R.update_qb(q.passer_player_id, q.q_epa, q.q_n)

    F = pd.DataFrame(rows)
    F["margin"] = F.home_score - F.away_score
    F["total"] = F.home_score + F.away_score
    return F


if __name__ == "__main__":
    F = build_features()
    F.to_parquet(f"{DATA}/game_features.parquet", index=False)
    print(F.shape)
    print(F[F.season == 2025].tail(3).T.head(30))
