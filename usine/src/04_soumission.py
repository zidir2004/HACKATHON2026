"""04_soumission.py — fabrique et contrôle un fichier de soumission à partir des
prédictions test d'une expérience (usine/experiences/oof/<EXP>_test.csv).

Usage : python -u usine/src/04_soumission.py --exp E01 --out usine/submissions/2_302.csv
"""
import argparse
import sys
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# 1. Prédictions test de l'expérience, réalignées sur l'ordre exact de test.csv
pred = pd.read_csv(U.OOF_DIR / f"{a.exp}_test.csv")
ids_test = pd.read_csv(U.TEST_CSV, usecols=[U.ID])
sub = ids_test.merge(pred, on=U.ID, how="left")

# 2. Écriture (sans en-tête) puis contrôle strict du format
U.ecrire_soumission(sub[U.ID], sub["prediction"], a.out)
taux = pd.read_csv(U.TRAIN_CSV, usecols=[U.TARGET])[U.TARGET].mean()
U.verifier_soumission(a.out, train_target_mean=taux)
