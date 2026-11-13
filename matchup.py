"""
matchup.py - "when offenses like A meet defenses like B, does the market misjudge it?"
For each game we look back at EARLIER games where the offense had a similar style to the home (or away)
offense AND the defense had a similar style to the opposing defense, and ask how many points that offense
scored relative to what Vegas implied. That average is the matchup signal. Test: does it predict Vegas's miss?
"""
import numpy as np, pandas as pd
from profiles import OFF_CORE, DEF_CORE, OFF_EXT, DEF_EXT

DATA = "data"


def load(extended=False):
    sch = pd.read_parquet(f"{DATA}/schedules.parquet")
    sch = sch[(sch.game_type == "REG")].copy()
    po = pd.read_parquet(f"{DATA}/off_profiles.parquet"); pdn = pd.read_parquet(f"{DATA}/def_profiles.parquet")
    oc = OFF_CORE + (OFF_EXT if extended else []); dc = DEF_CORE + (DEF_EXT if extended else [])
    ref = po.merge(sch[["game_id", "season"]], on="game_id"); refd = pdn.merge(sch[["game_id", "season"]], on="game_id")
    om, osd = ref[ref.season <= 2021][oc].mean(), ref[ref.season <= 2021][oc].std()
    dm, dsd = refd[refd.season <= 2021][dc].mean(), refd[refd.season <= 2021][dc].std()
    zo = po.set_index(["game_id", "team"])[oc].sub(om).div(osd).fillna(0)
    zd = pdn.set_index(["game_id", "team"])[dc].sub(dm).div(dsd).fillna(0)
    return sch, zo, zd


def pairings(sch, zo, zd):
    rows = []
    for _, g in sch.iterrows():
        gid = g.game_id
        if (gid, g.home_team) not in zo.index or (gid, g.away_team) not in zo.index: continue
        hi = (g.total_line + g.spread_line) / 2 if pd.notna(g.spread_line) else np.nan
        ai = (g.total_line - g.spread_line) / 2 if pd.notna(g.spread_line) else np.nan
        t = g.season * 100 + g.week
        # home offense vs away defense ; away offense vs home defense
        rows.append(dict(game_id=gid, t=t, side="home", off=g.home_team, dfn=g.away_team,
                         pts=g.home_score, implied=hi, ov=zo.loc[(gid, g.home_team)].values, dv=zd.loc[(gid, g.away_team)].values))
        rows.append(dict(game_id=gid, t=t, side="away", off=g.away_team, dfn=g.home_team,
                         pts=g.away_score, implied=ai, ov=zo.loc[(gid, g.away_team)].values, dv=zd.loc[(gid, g.home_team)].values))
    P = pd.DataFrame(rows)
    return P


def matchup_signal(P, K=50, test_seasons=(2022, 2023, 2024, 2025), placebo=False, seed=0):
    rng = np.random.default_rng(seed)
    OV, DV = np.vstack(P.ov.values), np.vstack(P.dv.values)
    T = P.t.values; OFF = P.off.values; DFN = P.dfn.values
    y = (P.pts - P.implied).values                      # points over market expectation
    if placebo:                                         # break the link between style and outcome
        perm = rng.permutation(len(P)); OV, DV = OV[perm], DV[perm]
    sig = np.full(len(P), np.nan)
    for t in sorted(set(T[(T // 100).astype(int).astype(int) >= min(test_seasons)])):
        if (t // 100) not in test_seasons: continue
        idx = np.where(T == t)[0]
        lib = np.where((T < t) & ~np.isnan(y))[0]
        for i in idx:
            keep = lib[(OFF[lib] != OFF[i]) & (DFN[lib] != DFN[i])]       # style, not team identity
            d = ((OV[keep] - OV[i]) ** 2).sum(1) + ((DV[keep] - DV[i]) ** 2).sum(1)
            nn = keep[np.argsort(d)[:K]]
            sig[i] = y[nn].mean()
    P = P.copy(); P["sig"] = sig
    return P


def ols(y, x):
    x, y = np.asarray(x, float), np.asarray(y, float); X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    se = np.sqrt((r @ r / (len(x) - 2)) / ((x - x.mean()) ** 2).sum())
    return b[1], b[1] / se


def evaluate(P, sch, label):
    h = P[P.side == "home"].set_index("game_id").sig; a = P[P.side == "away"].set_index("game_id").sig
    g = sch.set_index("game_id").join(pd.DataFrame({"sh": h, "sa": a})).dropna(subset=["sh", "sa", "home_score", "spread_line"])
    g["merr"] = g.home_score - g.away_score - g.spread_line; g["terr"] = g.home_score + g.away_score - g.total_line
    g["msig"] = g.sh - g.sa; g["tsig"] = g.sh + g.sa
    out = {}
    bm, tm = ols(g.merr, g.msig); bt, tt = ols(g.terr, g.tsig)
    per_m = "/".join(f"{ols(d.merr, d.msig)[0]:+.2f}" for _, d in g.groupby("season"))
    per_t = "/".join(f"{ols(d.terr, d.tsig)[0]:+.2f}" for _, d in g.groupby("season"))
    print(f"{label:34s} n={len(g)} | MARGIN slope {bm:+.2f} (t={tm:+.1f}) [{per_m}] | TOTAL slope {bt:+.2f} (t={tt:+.1f}) [{per_t}]")
    return g


if __name__ == "__main__":
    for ext in [False, True]:
        sch, zo, zd = load(extended=ext)
        P = pairings(sch, zo, zd)
        for K in [25, 50, 100]:
            S = matchup_signal(P, K=K); evaluate(S, sch, f"{'core+ext' if ext else 'core'} K={K}")
        S = matchup_signal(P, K=50, placebo=True); evaluate(S, sch, f"PLACEBO {'core+ext' if ext else 'core'} K=50")
