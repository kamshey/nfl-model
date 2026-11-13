"""
v11.py - fixes found while reviewing week 9:
  1. Missing weather on outdoor games is imputed with the historical outdoor average (was: calm, 0 mph / 70F).
  2. QB prior depends on career experience: low-history QBs are rated below average (from 2019+2021 data only).
  3. A game is 'low confidence' when either starter changed since his team's last game or has < 300 career dropbacks.
"""
import numpy as np, pandas as pd
import ratings_v2 as rv
from ratings import Ratings

OFFS_X = [0, 50, 150, 300, 600]
OFFS_Y = [-0.14, -0.085, -0.07, 0.0, 0.04]       # EPA/dropback vs league mean, from 2019+2021 (see analysis)


class Ratings11(Ratings):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.career = {}; self.season_idx = 0
    def new_season(self, years_gap=1):
        super().new_season(years_gap); self.season_idx += 1
    def update_qb(self, qid, e, n):
        super().update_qb(qid, e, n); self.career[qid] = self.career.get(qid, 0) + n
    def qb_rating(self, qid):
        warm = self.season_idx >= 1                        # 2018 is warm-up: career counts not yet meaningful
        if qid is None or qid != qid:
            return self.mu["pass"] + (OFFS_Y[0] if warm else -0.10), 0.0
        off = float(np.interp(self.career.get(qid, 0), OFFS_X, OFFS_Y)) if warm else -0.05
        e, n = self.qb.get(qid, [0.0, 0.0])
        return (e + self.qb_k * (self.mu["pass"] + off)) / (n + self.qb_k), n


def qb_flags(F):
    """low_conf per game from schedule starters + career dropbacks strictly before the game."""
    S = pd.read_parquet("data/schedules.parquet"); S = S[S.game_type == "REG"]
    pbp = pd.read_parquet("data/pbp.parquet", columns=["season", "week", "season_type", "qb_dropback", "passer_player_id"])
    d = pbp[(pbp.season_type == "REG") & (pbp.qb_dropback == 1) & pbp.passer_player_id.notna()]
    g = d.groupby(["passer_player_id", "season", "week"]).size().rename("n").reset_index()
    g["t"] = (g.season * 100 + g.week).astype("int64"); g = g.sort_values(["passer_player_id", "t"])
    g["cum"] = g.groupby("passer_player_id").n.cumsum()
    rows = []
    for side in ("home", "away"):
        x = S[["game_id", "season", "week", f"{side}_team", f"{side}_qb_id"]].set_axis(["game_id", "season", "week", "team", "qb"], axis=1); x["side"] = side
        rows.append(x)
    Q = pd.concat(rows).sort_values(["team", "season", "week"])
    Q["prev_qb"] = Q.groupby("team").qb.shift(1)
    Q["changed"] = (Q.qb != Q.prev_qb) & Q.prev_qb.notna() & Q.qb.notna()
    Q["t"] = (Q.season * 100 + Q.week).astype("int64")
    Qs = Q.dropna(subset=["qb"]).sort_values("t")
    m = pd.merge_asof(Qs, g[["passer_player_id", "t", "cum"]].sort_values("t"), left_on="t", right_on="t",
                      left_by="qb", right_by="passer_player_id", allow_exact_matches=False)
    Q = Q.merge(m[["game_id", "side", "cum"]], on=["game_id", "side"], how="left")
    Q["career"] = Q.cum.fillna(0)
    Q["low"] = Q.changed | (Q.career < 300) | Q.qb.isna()
    w = Q.pivot(index="game_id", columns="side", values=["low", "changed", "career"])
    out = pd.DataFrame({"low_conf": w["low"].any(axis=1), "qb_changed": w["changed"].any(axis=1),
                        "h_career": w[("career", "home")], "a_career": w[("career", "away")]}).reset_index()
    return F.merge(out, on="game_id", how="left")


def fix_weather(F):
    S = pd.read_parquet("data/schedules.parquet")[["game_id", "season", "roof", "wind", "temp"]]
    out = S[(~S.roof.isin(["dome", "closed"])) & (S.season <= 2021)]
    wmean, tmean = out.wind.mean(), out.temp.mean()
    miss = S[(~S.roof.isin(["dome", "closed"])) & S.wind.isna()].game_id
    F = F.copy(); idx = F.game_id.isin(miss)
    F.loc[idx, "wind"], F.loc[idx, "temp"] = wmean, tmean
    print(f"weather: imputed {int(idx.sum())} outdoor games with wind {wmean:.1f} mph / temp {tmean:.0f}F")
    return F


def build_v11():
    rv.Ratings = Ratings11
    F = rv.build_features(alpha=0.06, shrink=0.8, luck=False, qb_delta=True, injuries=False)
    return qb_flags(fix_weather(F))
