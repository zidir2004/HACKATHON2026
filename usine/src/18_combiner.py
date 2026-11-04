"""18_combiner.py — mélange final : moyenne des rangs à poids égaux + remise à l'échelle probabilité.

Usage (cwd = Hackathon, CPU, quelques secondes) :
  python -u usine/src/18_combiner.py --membres usine/experiences/oof/E80_test.csv ... --sortie usine/submissions/2_308.csv
         [--reference usine/submissions/2_306.csv ...] [--probas <csv test>] [--sauver-melange usine/experiences/oof/E84_test.csv]
Chaque membre = CSV avec en-tête id,prediction (fichiers oof/<EXP>_test.csv).
Étapes :
  1. lecture, tri par id, vérification que tous les membres ont les mêmes ids
  2. rangs normalisés (0..1) de chaque membre, moyenne à poids égaux -> ordre du mélange
  3. quantile mapping (même principe que 07_rangs_vers_proba.py) : on garde l'ordre du mélange
     et on lui attribue les valeurs triées de la moyenne des probabilités des membres
  4. écriture sans en-tête dans l'ordre de test.csv + contrôles : format strict (utils.verifier_soumission),
     Spearman entre membres, Spearman sortie vs mélange (= 1) et vs --reference, moyenne ≈ 0,036
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--membres", nargs="+", required=True)
ap.add_argument("--sortie", required=True)
ap.add_argument("--probas", nargs="+", default=None,
                help="fichiers dont la moyenne des probabilités donne l'échelle (défaut = --membres) ; "
                     "utile si un membre est un mélange de rangs sans probabilités (ex. E73)")
ap.add_argument("--reference", nargs="*", default=None, help="soumission(s) sans en-tête à comparer (ex. 2_306.csv)")
ap.add_argument("--sauver-melange", default=None, help="CSV id,prediction des rangs moyens (optionnel)")
a = ap.parse_args()

# 1. lecture
M = [pd.read_csv(f).sort_values(U.ID, ignore_index=True) for f in a.membres]
ids = M[0][U.ID].to_numpy()
for f, m in zip(a.membres, M):
    assert (m[U.ID].to_numpy() == ids).all(), f"ids différents : {f}"
    assert m["prediction"].notna().all(), f"NaN : {f}"
P = np.array([m["prediction"].to_numpy(dtype=float) for m in M])

Q = P
if a.probas:
    Qm = [pd.read_csv(f).sort_values(U.ID, ignore_index=True) for f in a.probas]
    for f, m in zip(a.probas, Qm):
        assert (m[U.ID].to_numpy() == ids).all() and m["prediction"].notna().all(), f
    Q = np.array([m["prediction"].to_numpy(dtype=float) for m in Qm])

# 2. moyenne des rangs à poids égaux
R = np.array([rankdata(p) / len(p) for p in P])
melange = R.mean(axis=0)

# 3. quantile mapping vers l'échelle des probabilités
ordre = np.argsort(melange, kind="mergesort")
pred = np.empty(len(ordre)); pred[ordre] = np.sort(Q.mean(axis=0))
rho = spearmanr(pred, melange).correlation
assert rho > 0.9999, rho

# 4. écriture (ordre de test.csv) + contrôles
test_ids = pd.read_csv(U.TEST_CSV, usecols=[U.ID])[U.ID].to_numpy()
s = pd.Series(pred, index=ids).reindex(test_ids)
assert s.notna().all()
U.ecrire_soumission(test_ids, s.to_numpy(), a.sortie)
if a.sauver_melange:
    pd.DataFrame({U.ID: ids, "prediction": melange}).to_csv(a.sauver_melange, index=False)
print("Spearman entre membres :")
for i in range(len(M)):
    print("  ", pathlib.Path(a.membres[i]).name, [round(float(spearmanr(P[i], P[j]).correlation), 4) for j in range(len(M))])
print(f"Spearman sortie vs mélange de rangs = {rho:.6f} ; moyenne = {pred.mean():.5f} (attendu ≈ 0,036)")
y_moy = pd.read_csv(U.ROOT / "train.csv", usecols=[U.TARGET])[U.TARGET].mean() if (U.ROOT / "train.csv").exists() else None
U.verifier_soumission(a.sortie, y_moy)
with open(a.sortie, "rb") as f:
    assert b"\r" not in f.read(), "CRLF détecté"
print("fins de ligne LF : OK")
for r in (a.reference or []):
    ref = pd.read_csv(r, header=None, names=[U.ID, "prediction"])
    assert (ref[U.ID].to_numpy() == test_ids).all()
    print(f"Spearman sortie vs {pathlib.Path(r).name} = {spearmanr(s.to_numpy(), ref['prediction']).correlation:.5f}")
