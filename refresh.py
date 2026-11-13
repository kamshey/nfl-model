"""One command to refresh everything: python refresh.py data.json
download data -> style profiles -> ratings features -> 2026 walk-forward -> player data -> data.json"""
import subprocess, sys
import nflreadpy
def run(*a):
    print(">>", *a, flush=True); subprocess.run([sys.executable, *a], check=True)
out = sys.argv[1] if len(sys.argv) > 1 else "data.json"
run("get_nfl_data.py"); run("profiles.py")
import v11, backtest2 as b2
F = v11.build_v11(); F.to_parquet("data/game_features_v11.parquet", index=False)
R = b2.walk(F, b2.M0 + b2.QB, b2.T0, seasons=[int(nflreadpy.get_current_season())])
R["low_conf"] = R.game_id.map(F.set_index("game_id").low_conf); R.to_parquet("data/wf2026_v11_internal.parquet", index=False)
run("players.py"); run("export_site_data.py", out)
