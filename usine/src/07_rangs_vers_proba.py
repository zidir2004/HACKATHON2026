"""07_rangs_vers_proba.py — remet un mélange de rangs à l'échelle des probabilités.

Rôle dans la chaîne 2_306 : DERNIÈRE étape. Transforme le mélange E33f (moyenne de rangs)
en fichier de soumission usine/submissions/2_306.csv.
Entrées  : usine/experiences/oof/E33f_test.csv (rangs du mélange)
           usine/experiences/oof/{E30,E07,E08,E32b}_test.csv (probabilités des modèles membres)
Sortie   : usine/submissions/2_306.csv (sans en-tête, colonnes id,prediction, ordre de test.csv)
Commande : python -u usine/src/07_rangs_vers_proba.py --exp E33f --membres E30 E07 E08 E32b --sortie usine/submissions/2_306.csv
           (CPU, quelques secondes)

Pourquoi : le sujet demande une prédiction « en probabilité ». Le mélange E09 moyenne des RANGS
(valeurs entre 0 et 1, moyenne 0,5), ce qui n'est pas une probabilité.
Comment : on garde EXACTEMENT l'ordre du mélange (donc le même Gini) et on lui attribue les valeurs
triées de la moyenne des probabilités des modèles membres (« quantile mapping »).
Résultat : même classement, valeurs de probabilités réalistes (moyenne ≈ 3,6 %).

Usage : python -u usine/src/07_rangs_vers_proba.py --exp E09 --membres E02 E07 E08 --sortie usine/submissions/2_303.csv
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

# === 1. Arguments ===
ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--membres", nargs="+", required=True)
ap.add_argument("--sortie", required=True)
a = ap.parse_args()

# === 2. Chargement : rangs du mélange + probabilités des membres, alignés par id ===
melange = pd.read_csv(U.OOF_DIR / f"{a.exp}_test.csv").sort_values(U.ID, ignore_index=True)
probas = [pd.read_csv(U.OOF_DIR / f"{m}_test.csv").sort_values(U.ID, ignore_index=True) for m in a.membres]
assert all((p[U.ID].values == melange[U.ID].values).all() for p in probas)
# Moyenne des probabilités des membres : fournit des VALEURS réalistes (moyenne ≈ 3,6 %),
# mais c'est le mélange de rangs qui fixe l'ORDRE (donc le Gini).
moy_proba = np.mean([p["prediction"].to_numpy() for p in probas], axis=0)

# === 3. Quantile mapping ===
# L'assuré classé k-ième par le mélange reçoit la k-ième plus petite probabilité moyenne
# mergesort = tri stable : les ex-aequo gardent l'ordre des id (résultat reproductible)
ordre = np.argsort(melange["prediction"].to_numpy(), kind="mergesort")
pred = np.empty(len(ordre))
pred[ordre] = np.sort(moy_proba)

# === 4. Contrôles puis écriture ===
# Contrôle : l'ordre est strictement conservé (corrélation de Spearman = 1 hors ex-aequo)
from scipy.stats import spearmanr
rho = spearmanr(pred, melange["prediction"]).correlation
print(f"Spearman mélange vs sortie = {rho:.6f} ; moyenne = {pred.mean():.4f} ; min {pred.min():.4f} max {pred.max():.4f}")
assert rho > 0.9999
# Écriture au format exigé, puis contrôle automatique (lignes, ids, ordre, NaN, [0,1])
U.ecrire_soumission(melange[U.ID], pred, a.sortie)
print("contrôle format :", U.verifier_soumission(a.sortie))
