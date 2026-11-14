"""Exporter (called by refresh.py): predicts the next unplayed week, explains every game, adds player estimates, writes data.json."""
import json, re, sys, os, urllib.request
import numpy as np, pandas as pd
from scipy.stats import norm
import nflreadpy, backtest2 as b2, matchup as mu, profiles as pf, similar
SEASON = int(nflreadpy.get_current_season())
from sklearn.ensemble import HistGradientBoostingRegressor

F = pd.read_parquet("data/game_features_v11.parquet")
S = pd.read_parquet("data/schedules.parquet"); S = S[(S.season == SEASON) & (S.game_type == "REG")].set_index("game_id")
if S[S.home_score.isna()].empty: sys.exit("No unplayed games left this season; nothing to predict.")
UP = int(S[S.home_score.isna()].week.min())
MC, TC = b2.M0 + b2.QB, b2.T0
# ---- 1. predictions: weeks 1..UP-1 from the walk-forward file, UP fitted on everything played ----
played = F[F.margin.notna() & F.spread_line.notna()]; tr = played[~((played.season == 2018) & (played.week < 4))]
te = F[(F.season == SEASON) & (F.week == UP) & F.margin.isna()].copy()
te["pm"], te["sig"] = b2.fitpred(tr, te, MC, "margin"); te["pt"], _ = b2.fitpred(tr, te, TC, "total")
past = pd.read_parquet("data/wf2026_v11_internal.parquet")
P = pd.concat([past[past.week <= UP], te]).drop_duplicates("game_id").set_index("game_id")
# ---- 2. style-matchup signal (forward profiles for the upcoming week: state after the last played game) ----
fut = pd.concat([pd.DataFrame({"game_id": g, "team": [r.home_team, r.away_team]}) for g, r in S[S.week == UP].iterrows()])
for nm, cols, f in [("off", pf.OFF_CORE + pf.OFF_EXT, "off_game_stats"), ("def", pf.DEF_CORE + pf.DEF_EXT, "def_game_stats")]:
    st = pd.read_parquet(f"data/{f}.parquet"); st = st[~st.game_id.isin(fut.game_id)]
    pd.concat([st, fut]).pipe(lambda d: pf.pregame_profiles(d, cols)).to_parquet(f"data/{nm}_profiles.parquet", index=False)
sch, zo, zd = mu.load(extended=True); PP = mu.pairings(sch, zo, zd); sh = {}
for K in (50, 100):
    X = mu.matchup_signal(PP, K=K, test_seasons=(SEASON,)).set_index(["game_id", "side"]).sig
    for k, v in X.items(): sh.setdefault(k, []).append(v)
sh = {k: float(np.mean(v)) for k, v in sh.items()}
# ---- 3. team style tags ----
o, d = similar.season_profiles(); od, dd = o.xs(SEASON, level="season"), d.xs(SEASON, level="season")
otag = {t: similar.describe(od.loc[t], o) for t in od.index}
dz = (dd - dd.mean()) / dd.std(); dtag = {t: ("leaky pass defense" if dz.loc[t, "epa_pass_allowed"] > 1 else "tough pass defense" if dz.loc[t, "epa_pass_allowed"] < -1 else "") +
    ("; " if abs(dz.loc[t, "epa_pass_allowed"]) > 1 and abs(dz.loc[t, "epa_rush_allowed"]) > 1 else "") +
    ("leaky run defense" if dz.loc[t, "epa_rush_allowed"] > 1 else "stout run defense" if dz.loc[t, "epa_rush_allowed"] < -1 else "") for t in dz.index}
# ---- 4. players: lite model (usage + game context), trained on 2018-2025 ----
ps = pd.read_parquet("data/player_dataset.parquet"); ps["is_RB"], ps["is_TE"] = (ps.position == "RB").astype(int), (ps.position == "TE").astype(int)
CTX = ["n_prior", "is_home", "team_total", "spread_for"]
REC = ["e_targets", "e_receptions", "e_receiving_yards", "e_receiving_tds", "e_tgt_share", "is_RB", "is_TE"]; RUS = ["e_carries", "e_rushing_yards", "e_rushing_tds", "e_car_share"]; PAS = ["e_attempts", "e_passing_yards", "e_passing_tds"]
SPEC = {"Rec": ("receptions", REC, lambda d: d.position.isin(["WR", "TE", "RB"]) & (d.e_targets >= 2), 1.6), "Rec yds": ("receiving_yards", REC, lambda d: d.position.isin(["WR", "TE", "RB"]) & (d.e_targets >= 2), 21.5),
        "Rush yds": ("rushing_yards", RUS, lambda d: (d.position == "RB") & (d.e_carries >= 4), 24.8), "Pass yds": ("passing_yards", PAS, lambda d: (d.position == "QB") & (d.e_attempts >= 15), 66.0)}
M = {}
for lab, (y, own, rows, _) in SPEC.items():
    t = ps[(ps.season < SEASON) & rows(ps) & (ps.n_prior >= 3)]
    M[lab] = HistGradientBoostingRegressor(loss="poisson", max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=60, l2_regularization=5.0, random_state=0).fit(t[CTX + own], t[y].clip(lower=0))
def est(rowdf, with_actual):
    out = {}
    for lab, (y, own, rows, miss) in SPEC.items():
        r = rowdf[rows(rowdf)]
        if len(r):
            for i, p in zip(r.index, M[lab].predict(r[CTX + own])): out.setdefault(i, []).append([lab, round(float(p), 1), miss, int(rowdf.loc[i, y]) if with_actual else None])
    return out
def pick(rowdf, n_rec=3):
    r = rowdf.copy(); keep = []
    q = r[r.position == "QB"].sort_values("e_attempts", ascending=False).head(1); keep += list(q.index)
    rb = r[r.position == "RB"].sort_values("e_carries", ascending=False).head(1); keep += list(rb.index)
    rc = r[(r.position != "QB") & ~r.index.isin(keep)].sort_values("e_targets", ascending=False).head(n_rec); keep += list(rc.index)
    return r.loc[keep]
props = {}
pl = ps[(ps.season == SEASON)]
for gid, g in pl.groupby("game_id"):                                         # past games of SEASON: estimate vs actual
    for team, tg in g.groupby("team"):
        top = pick(tg); e = est(top, True)
        props.setdefault(gid, []).extend([dict(t=team, n=top.loc[i, "player_display_name"], p=top.loc[i, "position"], s=v) for i, v in e.items()])
hist = ["targets", "receptions", "receiving_yards", "receiving_tds", "carries", "rushing_yards", "rushing_tds", "attempts", "passing_yards", "passing_tds", "tgt_share", "car_share"]
ps = ps.sort_values(["player_id", "season", "week"]); last = ps.groupby("player_id").tail(1).set_index("player_id")
cur = ps.groupby("player_id")[hist].apply(lambda x: x.ewm(alpha=0.3).mean().iloc[-1]); cur.columns = ["e_" + c for c in hist]
cur = cur.join(last[["player_display_name", "position", "team", "season", "week", "is_RB", "is_TE"]]); cur["n_prior"] = ps.groupby("player_id").size()
inj = pd.read_parquet("data/injuries.parquet"); inj = inj[inj.season == SEASON]; inj = inj[inj.week == inj.week.max()]
out_ids = set(inj[inj.report_status.isin(["Out", "Doubtful"])].gsis_id); q_ids = set(inj[inj.report_status == "Questionable"].gsis_id)
tlast = ps[ps.season == SEASON].groupby("team").week.max()
for gid, g in S[S.week == UP].iterrows():
    for team, home in [(g.home_team, 1), (g.away_team, 0)]:
        c = cur[(cur.team == team) & (cur.season == SEASON) & (cur.week >= tlast.get(team, 0) - 1) & ~cur.index.isin(out_ids)].copy()
        c["is_home"], c["team_total"] = home, (g.total_line + g.spread_line) / 2 if home else (g.total_line - g.spread_line) / 2
        c["spread_for"] = g.spread_line if home else -g.spread_line
        qid = g.home_qb_id if home else g.away_qb_id; c = c[(c.position != "QB") | (c.index == qid)]
        top = pick(c); e = est(top, False)
        props.setdefault(gid, []).extend([dict(t=team, n=top.loc[i, "player_display_name"] + (" (Q)" if i in q_ids else ""), p=top.loc[i, "position"], s=v) for i, v in e.items()])
# ---- 4b. kickoff weather forecasts (Open-Meteo, free, no key) for upcoming outdoor games ----
COORDS = {"ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62), "BUF": (42.77, -78.79), "CAR": (35.23, -80.85), "CHI": (41.86, -87.62), "CIN": (39.10, -84.52), "CLE": (41.51, -81.70),
          "DAL": (32.75, -97.09), "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06), "HOU": (29.68, -95.41), "IND": (39.76, -86.16), "JAX": (30.32, -81.64), "KC": (39.05, -94.48),
          "LA": (33.95, -118.34), "LAC": (33.95, -118.34), "LV": (36.09, -115.18), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26), "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07),
          "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02), "SEA": (47.60, -122.33), "SF": (37.40, -121.97), "TB": (27.98, -82.50), "TEN": (36.17, -86.77), "WAS": (38.91, -76.86)}
ANGLES = [dict(id="wind", name="UNDER in windy games", stat="UNDER hit 58% of 516 outdoor games with 12+ mph wind, 2012-2025"), dict(id="early", name="Underdogs in weeks 1-3", stat="Underdogs covered 55% of 328 games in weeks 1-3, 2018-2025")]
def forecast(lat, lon):
    if os.environ.get("FAKE_WIND"): return {pd.Timestamp(2026, 1, 1, tz="UTC") + pd.Timedelta(hours=i): (14.0, 41.0, 20) for i in range(24 * 400)}
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=wind_speed_10m,temperature_2m,precipitation_probability&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=UTC&forecast_days=16"
    try:
        with urllib.request.urlopen(url, timeout=20) as f: j = json.load(f)["hourly"]
        return {pd.Timestamp(t, tz="UTC"): (w, tp, pp) for t, w, tp, pp in zip(j["time"], j["wind_speed_10m"], j["temperature_2m"], j["precipitation_probability"])}
    except Exception as e:
        print("forecast failed:", e); return {}
FC = {}
for gid, g in S[(S.week == UP) & S.home_score.isna() & ~S.roof.isin(["dome", "closed"])].iterrows():
    if pd.isna(g.gametime) or g.home_team not in COORDS: continue
    v = forecast(*COORDS[g.home_team]).get(pd.Timestamp(f"{g.gameday} {g.gametime}", tz="America/New_York").tz_convert("UTC").floor("h"))
    if v and v[0] is not None: FC[gid] = dict(wind=float(v[0]), temp=round(float(v[1])), pop=v[2])
# ---- 5. per-game tiers + reasons ----
Q = pd.concat([S[["week", "home_team", "home_qb_id", "home_qb_name"]].set_axis(["week", "team", "qb", "name"], axis=1).assign(gid=S.index, side="h"),
               S[["week", "away_team", "away_qb_id", "away_qb_name"]].set_axis(["week", "team", "qb", "name"], axis=1).assign(gid=S.index, side="a")]).sort_values(["team", "week"])
Q["chg"] = (Q.qb != Q.groupby("team").qb.shift(1)) & Q.groupby("team").qb.shift(1).notna(); Qi = Q.set_index(["gid", "side"])
def line(m, h, a): return f"{h} {m:.1f}" if m < 0 else f"{a} {-m:.1f}" if m > 0 else "pick'em"
games = []
for gid, r in P.sort_values(["week"]).iterrows():
    s = S.loc[gid]; h, a = r.home_team, r.away_team; fin = [int(s.away_score), int(s.home_score)] if pd.notna(s.home_score) else None
    nl = pd.isna(r.spread_line); d_ = 0.0 if nl else r.pm - r.spread_line; ms = (sh.get((gid, "home"), np.nan) - sh.get((gid, "away"), np.nan)); ms = None if np.isnan(ms) else ms
    rat = int(np.sign(d_)) if abs(d_) >= 3 else 0; mus = int(np.sign(ms)) if ms is not None and abs(ms) >= 1 else 0
    qbn = []
    for side, tm in (("h", h), ("a", a)):
        q = Qi.loc[(gid, side)]; car = r.get("h_career" if side == "h" else "a_career", 0)
        if q.chg: qbn.append(f"{tm} starts {q['name']} (a change from last game).")
        elif q.qb in out_ids or q.qb in q_ids: qbn.append(f"{tm}'s starting QB {q['name']} is on the latest injury report.")
        elif pd.isna(q.qb) or car < 300: qbn.append(f"{tm} QB {q['name']} has only {int(car)} career dropbacks.")
    side_ = mus; conflict = rat and mus and rat != mus
    tier, lean, pc = "none", None, 0.5; why = []
    why.append(["info", f"Ratings project {h} {r.h_exp_pts:.0f} points and {a} {r.a_exp_pts:.0f}. Model line {line(-r.pm, h, a)}" + (", no Vegas line posted yet." if nl else f", Vegas {line(-r.spread_line, h, a)}.")])
    ue = {"pass": r.h_exp_pass - r.a_exp_pass, "rush": r.h_exp_rush - r.a_exp_rush}; k = max(ue, key=lambda x: abs(ue[x]))
    if rat: why.append(["for" if rat == mus else "info", f"The ratings like {h if rat > 0 else a} by {abs(d_):.1f} more points than Vegas does; the biggest edge is the {k} game, where {h if ue[k] > 0 else a} grades better."])
    if abs(r.h_qb - r.a_qb) > 0.08: why.append(["info", f"QB edge goes to {h if r.h_qb > r.a_qb else a} (rating {max(r.h_qb, r.a_qb):+.2f} vs {min(r.h_qb, r.a_qb):+.2f} EPA per dropback)."])
    if ms is not None: why.append(["for" if mus and mus == side_ else "info", f"Style matchup: teams that play like {h}'s offense against defenses like {a}'s have scored {sh[(gid, 'home')]:+.1f} points vs market expectation; {a}'s offense against a {h}-style defense {sh[(gid, 'away')]:+.1f}."])
    why.append(["info", f"{h} offense: {otag.get(h, '')}. {a} offense: {otag.get(a, '')}. {h} defense: {dtag.get(h) or 'average'}; {a} defense: {dtag.get(a) or 'average'}."])
    why += [["info", x] for x in qbn]
    if conflict: why.append(["info", "The ratings see this game the other way, so the lean rests on the style matchup."])
    if abs(r.rest_diff) >= 3: why.append(["info", f"{h if r.rest_diff > 0 else a} has {abs(int(r.rest_diff))} more days of rest."])
    if r.wind >= 15: why.append(["info", f"Wind around {r.wind:.0f} mph tends to hold scoring down."])
    dt = 0.0 if pd.isna(r.total_line) else r.pt - r.total_line; tot = int(np.sign(dt)) if abs(dt) >= 3 else 0
    if not pd.isna(r.total_line): why.append(["info", f"Model total {r.pt:.1f} vs Vegas {r.total_line:.1f}."])
    ko = pd.Timestamp(f"{s.gameday} {s.gametime}", tz="America/New_York").tz_convert("UTC") if pd.notna(s.gametime) else pd.NaT
    wx = FC.get(gid); picks = []
    if wx: why.append(["info", f"Kickoff forecast: wind {wx['wind']:.0f} mph, {wx['temp']}°F."])
    if fin is None and wx and wx["wind"] >= 12 and not pd.isna(r.total_line) and s.roof not in ("dome", "closed"):
        picks.append(dict(rule="wind", mk="Total", t=f"UNDER {r.total_line:g}", conf=2, note=f"Forecast wind {wx['wind']:.0f} mph. UNDER hit 58% of 516 games with 12+ mph wind (2012-2025)."))
    if fin is None and r.week <= 3 and not nl:
        picks.append(dict(rule="early", mk="Spread", t=f"{a if r.spread_line > 0 else h} +{abs(r.spread_line):g}", conf=1, note="Underdogs covered 55% of 328 games in weeks 1-3 (2018-2025)."))
    games.append(dict(wx=wx, picks=picks, kou=None if pd.isna(ko) else ko.isoformat(), id=gid, wk=int(r.week), away=a, home=h, spread=None if nl else float(r.spread_line), total=None if pd.isna(r.total_line) else float(r.total_line), pm=round(float(r.pm), 1), pt=round(float(r.pt), 1), tier=tier, lean=lean,
                      tot=0, pc=round(pc * 100, 1), tgap=round(float(dt), 1), ms=None if ms is None else round(float(ms), 1), final=fin, qb=[s.away_qb_name, s.home_qb_name], why=why, props=props.get(gid, []), day=str(s.gameday), ko=f"{s.gameday} {s.gametime}"))
# ---- archive: lock each pick once, before kickoff; score after the game ----
ARCH = []
try:
    with urllib.request.urlopen(os.environ["ARCHIVE_URL"], timeout=20) as f: ARCH = json.load(f)
except Exception as e: print("no previous archive:", e)
have = {x["key"] for x in ARCH}; now = pd.Timestamp.now(tz="UTC")
for g in games:
    for p in g["picks"]:
        key = f"{g['id']}|{p['rule']}"
        if key not in have and g["kou"] and pd.Timestamp(g["kou"]) > now:
            ARCH.append(dict(key=key, gid=g["id"], wk=g["wk"], rule=p["rule"], text=f"{p['t']} ({g['away']} at {g['home']})", team=p["t"].split()[0], spread=g["spread"], total=g["total"], locked=now.isoformat(), res=None))
for x in ARCH:
    if x["res"] is None and x["gid"] in S.index and pd.notna(S.loc[x["gid"]].home_score):
        z = S.loc[x["gid"]]; mg = z.home_score - z.away_score
        if x["rule"] == "wind": d = x["total"] - (z.home_score + z.away_score)
        else: d = (mg - x["spread"]) * (1 if x["team"] == z.home_team else -1)
        x["res"] = "P" if d == 0 else "W" if (d > 0 if x["rule"] == "wind" else d > 0) else "L"
out = sys.argv[1] if len(sys.argv) > 1 else "data.json"
json.dump(ARCH, open(os.path.join(os.path.dirname(out) or ".", "archive.json"), "w"), separators=(",", ":"))
json.dump(dict(games=games, angles=ANGLES, archive=ARCH, up=UP, season=SEASON, generated=pd.Timestamp.now(tz="UTC").isoformat()), open(out, "w"), separators=(",", ":"))
t = pd.Series([g["tier"] for g in games if g["wk"] == UP]).value_counts().to_dict(); print("upcoming week", UP, "| games", len([g for g in games if g["wk"] == UP]), "| tiers", t, "| wrote", out)
