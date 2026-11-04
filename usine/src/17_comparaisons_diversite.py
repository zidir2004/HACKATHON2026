"""17_comparaisons_diversite.py — piste B, diversité de réseaux (E50–E58) : comparaisons appariées
complémentaires (réseaux entre eux, corrélations de rang OOF entre réseaux et avec E09)."""
import importlib, pathlib, sys
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
T = importlib.import_module("03_train")
L = lambda e: pd.read_csv(U.OOF_DIR / f"{e}_oof.csv").sort_values(U.ID, ignore_index=True)["oof"]
noms = ["E09", "E32c", "E51c", "E52"]
print("Spearman OOF :\n", pd.DataFrame({n: L(n) for n in noms}).corr("spearman").round(4), flush=True)
for exp, ref in [("E51c", "E32c"), ("E52", "E32c")]:
    g = pd.read_csv(U.OOF_DIR / f"{exp}_folds_detail.csv").sort_values("fold")["gini"].to_numpy()
    c = T.comparer(exp, ref, g)
    print(f"{exp} vs {ref} : Δ {c['d_moy']:+.5f} | {c['folds_gagnes']} | seuil 2se {c['seuil_2se']} -> {c['verdict']}", flush=True)
