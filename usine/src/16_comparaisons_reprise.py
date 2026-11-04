"""16_comparaisons_reprise.py — reprise du DAE (dae_reprise.pt, 1320 époques au total) :
comparaisons appariées du nouveau réseau et du nouveau mélange contre leurs équivalents avec le DAE d'origine."""
import importlib, pathlib, sys
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
T = importlib.import_module("03_train")
for exp, ref in [("E40", "E32b"), ("E40c", "E32c"), ("E41", "E33f"), ("E41", "E09")]:
    g = pd.read_csv(U.OOF_DIR / f"{exp}_folds_detail.csv").sort_values("fold")["gini"].to_numpy()
    c = T.comparer(exp, ref, g)
    print(f"{exp} vs {ref} : Δ {c['d_moy']:+.5f} | {c['folds_gagnes']} | seuil 2se {c['seuil_2se']} | "
          f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} -> {c['verdict']}", flush=True)
