"""similar.py - "which teams, in recent history, played like this team?"  (offense style, season-level)"""
import numpy as np, pandas as pd
from profiles import OFF_CORE, DEF_CORE

def season_profiles():
    sch = pd.read_parquet("data/schedules.parquet")[["game_id", "season"]]
    off = pd.read_parquet("data/off_game_stats.parquet").merge(sch, on="game_id")
    dfn = pd.read_parquet("data/def_game_stats.parquet").merge(sch, on="game_id")
    return (off.groupby(["season", "team"])[OFF_CORE].mean(), dfn.groupby(["season", "team"])[DEF_CORE].mean())

def nearest(prof, key, k=3, exclude_same_team=True):
    z = (prof - prof.mean()) / prof.std()
    d = ((z - z.loc[key]) ** 2).sum(axis=1) ** 0.5
    d = d.drop(key)
    if exclude_same_team: d = d[[i[1] != key[1] for i in d.index]]
    return d.nsmallest(k)

def describe(row, prof):
    z = (row - prof.mean()) / prof.std(); tags = []
    if z.pass_rate_neutral > 1: tags.append("pass-heavy")
    if z.pass_rate_neutral < -1: tags.append("run-heavy")
    if z.plays_pg > 1: tags.append("fast pace")
    if z.plays_pg < -1: tags.append("slow pace")
    if z.deep_rate > 1: tags.append("deep shots")
    if z.shotgun_rate > 1: tags.append("shotgun-heavy")
    if z.epa_pass > 1: tags.append("elite passing efficiency")
    return ", ".join(tags) or "balanced"

if __name__ == "__main__":
    o, d = season_profiles()
    cur = o.xs(2026, level="season"); print(f"2026 so far: {len(cur)} teams\n")
    hist = o[o.index.get_level_values("season") < 2026]
    z = (o - o.mean()) / o.std()
    picks = {"most pass-heavy": cur.pass_rate_neutral.idxmax(), "most run-heavy": cur.pass_rate_neutral.idxmin(),
             "fastest pace": cur.plays_pg.idxmax(), "most deep shots": cur.deep_rate.idxmax()}
    for label, team in picks.items():
        key = (2026, team)
        r = o.loc[key]
        print(f"{label}: {team} 2026  [{describe(r, o)}]  neutral pass rate {r.pass_rate_neutral:.0%}, {r.plays_pg:.0f} plays/g, deep {r.deep_rate:.0%}, EPA/dropback {r.epa_pass:+.2f}")
        nn = nearest(o[o.index.get_level_values("season") <= 2026], key, k=3)
        for (s, t), dist in nn.items():
            rr = o.loc[(s, t)]
            print(f"     ~ {t} {s}  (distance {dist:.1f})  pass rate {rr.pass_rate_neutral:.0%}, {rr.plays_pg:.0f} plays/g, deep {rr.deep_rate:.0%}, EPA/dropback {rr.epa_pass:+.2f}")
        print()
