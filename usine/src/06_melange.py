"""06_melange.py — mélange de modèles sur les prédictions out-of-fold (OOF).

Rôle dans la chaîne 2_306 : utilisé DEUX fois.
  a) E32c = moyenne des 3 graines du réseau sur DAE (E32b, E32b_g1, E32b_g2) -> réduit la variance du réseau ;
  b) E33f = moyenne de E30 (LightGBM) + E07 (XGBoost) + E08 (CatBoost) + E32c (réseau DAE) -> modèle final.
Entrées  : usine/experiences/oof/<membre>_oof.csv et <membre>_test.csv pour chaque membre
Sorties  : usine/experiences/oof/<EXP>_oof.csv, <EXP>_test.csv, <EXP>_folds_detail.csv
           + une ligne dans usine/experiences/registre.csv (+ comparaison appariée vs --ref)
Commandes 2_306 (CPU, < 1 min chacune) :
  python -u usine/src/06_melange.py --exp E32c --membres E32b E32b_g1 E32b_g2 --ref E32b
  python -u usine/src/06_melange.py --exp E33f --membres E30 E07 E08 E32c --ref E31c

Usage générique : python -u usine/src/06_melange.py --exp E09 --membres E02 E07 E08 --ref E02

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
T = importlib.import_module("03_train")   # réutilise comparer() (comparaison appariée fold par fold)

# === 1. Arguments ===

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--membres", nargs="+", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--version", default="v1", help="version de données des membres (registre)")
a = ap.parse_args()

# === 2. Chargement des prédictions OOF et test des membres (alignées par id) ===
oofs = [pd.read_csv(U.OOF_DIR / f"{m}_oof.csv").sort_values(U.ID, ignore_index=True) for m in a.membres]
tests = [pd.read_csv(U.OOF_DIR / f"{m}_test.csv").sort_values(U.ID, ignore_index=True) for m in a.membres]
y, fold, ids = oofs[0][U.TARGET].to_numpy(), oofs[0]["fold"].to_numpy(), oofs[0][U.ID]
# === 3. Passage en rangs ===
# Moyenne des RANGS plutôt que des probabilités : le Gini ne dépend que de l'ordre, et les modèles
# n'ont pas la même échelle (un réseau et un CatBoost ne sont pas calibrés pareil). Les rangs mettent
# chaque modèle sur la même échelle [0,1] et donnent à chacun le même poids réel.
R = np.column_stack([rankdata(o["oof"]) / len(o) for o in oofs])
Rt = np.column_stack([rankdata(t["prediction"]) / len(t) for t in tests])

# Diagnostic : plus deux membres sont décorrélés, plus le mélange a de chances de gagner
print("corrélation des rangs OOF :\n", pd.DataFrame(R, columns=a.membres).corr().round(4))
for i, m in enumerate(a.membres):
    print(f"  {m} seul : gini OOF {U.gini(y, R[:, i]):.5f}")


def ginis_folds(p):
    """Gini de chaque fold (même découpage que tous les modèles) -> liste de 5 valeurs."""
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

# === 4. Mélange retenu : poids ÉGAUX ===
# Les poids égaux ne sont pas appris sur l'OOF, donc le score CV reste honnête (pas de sur-ajustement).
w = np.ones(len(a.membres)) / len(a.membres)
p, pt = R @ w, Rt @ w
gs = ginis_folds(p)
print(f"  => {a.exp} poids égaux : gini CV {np.mean(gs):.5f} ± {np.std(gs):.5f} | OOF {U.gini(y, p):.5f}")

# === 5. Sauvegardes + comparaison appariée + registre ===
# Le mélange est sauvegardé au même format qu'un modèle : il peut lui-même servir de membre
# (E32c entre dans E33f) et être converti en soumission par 07_rangs_vers_proba.py.
pd.DataFrame({U.ID: ids, U.TARGET: y, "oof": p, "fold": fold}).to_csv(U.OOF_DIR / f"{a.exp}_oof.csv", index=False)
pd.DataFrame({U.ID: tests[0][U.ID], "prediction": pt}).to_csv(U.OOF_DIR / f"{a.exp}_test.csv", index=False)
pd.DataFrame({"n_arbres": "n/a", "fold": range(5), "gini": gs, "duree_s": 0}).to_csv(
    U.OOF_DIR / f"{a.exp}_folds_detail.csv", index=False)
# Comparaison appariée : même fold, même individus -> on compare Gini(exp) - Gini(ref) fold par fold
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
