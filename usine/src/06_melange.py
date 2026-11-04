"""06_melange.py — mélange de modèles sur les prédictions out-of-fold (OOF).

Usage : python -u usine/src/06_melange.py --exp E09 --membres E02 E07 E08 --ref E02

Méthode : moyenne des RANGS (normalisés dans [0,1]) des prédictions de chaque membre, poids ÉGAUX
(aucun poids appris -> pas de biais d'optimisation sur l'OOF). Les rangs OOF sont calculés sur
tout l'OOF : le Gini par fold n'en dépend pas (seul l'ordre intra-fold compte).
Pour information seulement, une grille de poids est évaluée sur l'OOF (optimiste, non retenue).
Sauvegarde oof/<EXP>_oof.csv, _test.csv (rangs du test dans [0,1]), _folds_detail.csv,
comparaison appariée vs ref, ligne au registre.
"""
import argparse
import itertools
import sys
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
import importlib
T = importlib.import_module("03_train")

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--membres", nargs="+", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--version", default="v1", help="version de données des membres (registre)")
a = ap.parse_args()

oofs = [pd.read_csv(U.OOF_DIR / f"{m}_oof.csv").sort_values(U.ID, ignore_index=True) for m in a.membres]
tests = [pd.read_csv(U.OOF_DIR / f"{m}_test.csv").sort_values(U.ID, ignore_index=True) for m in a.membres]
y, fold, ids = oofs[0][U.TARGET].to_numpy(), oofs[0]["fold"].to_numpy(), oofs[0][U.ID]
R = np.column_stack([rankdata(o["oof"]) / len(o) for o in oofs])
Rt = np.column_stack([rankdata(t["prediction"]) / len(t) for t in tests])

print("corrélation des rangs OOF :\n", pd.DataFrame(R, columns=a.membres).corr().round(4))
for i, m in enumerate(a.membres):
    print(f"  {m} seul : gini OOF {U.gini(y, R[:, i]):.5f}")


def ginis_folds(p):
    return [U.gini(y[fold == k], p[fold == k]) for k in range(U.N_FOLDS)]


# Info : grille de poids (pas 0.1) -> optimiste car choisie sur l'OOF
best = None
for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(a.membres)):
    if abs(sum(w) - 1) > 1e-9:
        continue
    g = U.gini(y, R @ np.array(w))
    if best is None or g > best[0]:
        best = (g, w)
print(f"  [info] meilleurs poids sur OOF {np.round(best[1], 1)} -> gini OOF {best[0]:.5f} (biais optimiste)")

w = np.ones(len(a.membres)) / len(a.membres)
p, pt = R @ w, Rt @ w
gs = ginis_folds(p)
print(f"  => {a.exp} poids égaux : gini CV {np.mean(gs):.5f} ± {np.std(gs):.5f} | OOF {U.gini(y, p):.5f}")

pd.DataFrame({U.ID: ids, U.TARGET: y, "oof": p, "fold": fold}).to_csv(U.OOF_DIR / f"{a.exp}_oof.csv", index=False)
pd.DataFrame({U.ID: tests[0][U.ID], "prediction": pt}).to_csv(U.OOF_DIR / f"{a.exp}_test.csv", index=False)
pd.DataFrame({"n_arbres": "n/a", "fold": range(5), "gini": gs, "duree_s": 0}).to_csv(
    U.OOF_DIR / f"{a.exp}_folds_detail.csv", index=False)
c = T.comparer(a.exp, a.ref, gs)
print(f"  apparié vs {a.ref} : Δmoy={c['d_moy']:+.5f} folds gagnés {c['folds_gagnes']} "
      f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} -> {c['verdict']}")
U.ajouter_registre(dict(
    id=a.exp, donnees_version=a.version, modele="melange_rangs", verdict=c["verdict"], delta_vs_ref=c["d_moy"],
    changement=f"moyenne des rangs à poids égaux de {'+'.join(a.membres)}",
    score_cv_moyen=float(np.mean(gs)), score_cv_std=float(np.std(gs)),
    commentaire=(f"gini OOF {U.gini(y, p):.5f}; apparié vs {a.ref}: Δmoy {c['d_moy']:+.5f}, folds gagnés "
                 f"{c['folds_gagnes']}, Δ/fold {[c[f'd_fold{k}'] for k in range(5)]}; poids OOF-optimaux "
                 f"{np.round(best[1], 1).tolist()} -> {best[0]:.5f} (non retenus, optimistes)")))
