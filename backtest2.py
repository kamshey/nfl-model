"""
backtest2.py - ablation: does each upgrade help the STANDALONE model (no Vegas input)?
Plus the key test: does the model carry information beyond the Vegas line?
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import ratings_v2 as rv

TEST = [2022, 2023, 2024, 2025]
M0 = ["h_exp_pts", "a_exp_pts", "h_exp_pass", "a_exp_pass", "h_exp_rush", "a_exp_rush", "h_qb", "a_qb", "rest_diff", "div_game", "neutral"]
T0 = ["h_exp_pts", "a_exp_pts", "h_exp_pass", "a_exp_pass", "h_exp_rush", "a_exp_rush", "dome", "wind", "temp"]
QB = ["h_qb_delta", "a_qb_delta"]
INJ = ["inj_off_diff", "inj_def_diff"]
TINJ = ["h_inj_off", "a_inj_off", "h_inj_def", "a_inj_def"]


def fitpred(tr, te, cols, y, alpha=30.0):
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(tr[cols], tr[y])
    return m.predict(te[cols]), (tr[y] - m.predict(tr[cols])).std()


def walk(F, mcols, tcols, seasons=TEST):
    played = F[F.margin.notna() & F.spread_line.notna()]
    out = []
    for s in seasons:
        for wk in sorted(F[(F.season == s) & (F.game_type == "REG")].week.unique()):
            te = F[(F.season == s) & (F.week == wk) & F.margin.notna() & F.spread_line.notna()]
            if te.empty: continue
            bf = played[(played.season < s) | ((played.season == s) & (played.week < wk))]
            tr = bf[~((bf.season == 2018) & (bf.week < 4))]
            t = te.copy()
            t["pm"], t["sig"] = fitpred(tr, te, mcols, "margin")
            t["pt"], _ = fitpred(tr, te, tcols, "total")
            out.append(t)
    return pd.concat(out)


def ols_slope(y, x):
    x = np.asarray(x); y = np.asarray(y); X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    se = np.sqrt((r @ r / (len(x) - 2)) / ((x - x.mean()) ** 2).sum())
    return b[1], b[1] / se


def summarize(R, name):
    R = R[(R.margin != 0)]
    win = (R.margin > 0).astype(int)
    acc = ((R.pm > 0) == (R.margin > 0)).mean()
    mae = np.abs(R.margin - R.pm).mean()
    brier = ((norm.cdf(R.pm / R.sig) - win) ** 2).mean()
    tmae = np.abs(R.total - R.pt).mean()
    # information beyond the line: (actual - spread) regressed on (model - spread); 0 = nothing extra, 1 = model beats the line outright
    b, t = ols_slope(R.margin - R.spread_line, R.pm - R.spread_line)
    bt, tt = ols_slope(R.total - R.total_line, R.pt - R.total_line)
    per = []
    for s in TEST:
        d = R[R.season == s]; per.append(ols_slope(d.margin - d.spread_line, d.pm - d.spread_line)[0])
    return dict(model=name, acc=acc, margin_MAE=mae, brier=brier, total_MAE=tmae,
                info_slope=b, info_t=t, tot_slope=bt, tot_t=tt, per_season="/".join(f"{x:+.2f}" for x in per))


if __name__ == "__main__":
    results, preds = [], {}
    base = pd.read_parquet("data/game_features.parquet")            # v1 ratings from earlier
    base = base[base.season <= 2025]
    variants = {
        "v0  original ratings": (base, M0, T0),
    }
    F1 = rv.build_features(luck=True, qb_delta=False, injuries=False); F1 = F1[F1.season <= 2025]
    F2 = rv.build_features(luck=True, qb_delta=True, injuries=True);   F2 = F2[F2.season <= 2025]
    variants["v1  + luck-adjusted"] = (F1, M0, T0)
    variants["v2  + QB-change feature"] = (F2, M0 + QB, T0)
    variants["v3  + injuries"] = (F2, M0 + QB + INJ, T0 + TINJ)
    for name, (F, mc, tc) in variants.items():
        R = walk(F, mc, tc); preds[name] = R
        results.append(summarize(R, name))
    ref = preds["v0  original ratings"]; ref = ref[ref.margin != 0]
    vm = np.abs(ref.margin - ref.spread_line).mean()
    vt = np.abs(ref.total - ref.total_line).mean()
    vacc = ((ref.spread_line > 0) == (ref.margin > 0)).mean()
    df = pd.DataFrame(results).set_index("model")
    print(f"Vegas for reference: winner acc {vacc:.3f} | margin MAE {vm:.2f} | total MAE {vt:.2f}\n")
    print(df.round(3).to_string())
    pd.to_pickle(preds, "data/ablation_preds.pkl")
