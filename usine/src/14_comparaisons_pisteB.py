"""14_comparaisons_pisteB.py — comparaisons appariées complémentaires de la piste B (DAE + réseau).

E32b vs E32 : effet du passage « early stopping par fold » -> « moyenne des époques 1..6 ».
E33f/E33e vs E09 : gain total des mélanges avec réseau par rapport à la référence historique.
"""
import importlib, pathlib, sys
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
T = importlib.import_module("03_train")
for exp, ref in [("E32b", "E32"), ("E32c", "E32b"), ("E33e", "E09"), ("E33f", "E09"), ("E33f", "E33e")]:
    g = pd.read_csv(U.OOF_DIR / f"{exp}_folds_detail.csv").sort_values("fold")["gini"].to_numpy()
    c = T.comparer(exp, ref, g)
    print(f"{exp} vs {ref} : Δ {c['d_moy']:+.5f} | {c['folds_gagnes']} | seuil 2se {c['seuil_2se']} | "
          f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} -> {c['verdict']}", flush=True)
