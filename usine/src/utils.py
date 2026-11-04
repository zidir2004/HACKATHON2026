"""utils.py — fonctions communes de l'usine ML (hackathon M2, Porto Seguro).

Contenu :
  1. Chemins du projet
  2. Métrique officielle : gini = 2*AUC - 1
  3. Chargement des données (par version)
  4. Protocole de validation : création / lecture des folds (fichier unique pour tout le projet)
  5. Registre des expériences
  6. Écriture + contrôle du format de soumission

Tous les chemins sont relatifs au dossier Hackathon (cwd du lanceur).
"""
from __future__ import annotations

import csv
import datetime as dt
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# 1. Chemins
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]          # dossier Hackathon
USINE = ROOT / "usine"
TRAIN_CSV, TEST_CSV = ROOT / "train.csv", ROOT / "test.csv"
EXP_DIR = USINE / "experiences"
OOF_DIR = EXP_DIR / "oof"
FOLDS_CSV = EXP_DIR / "folds.csv"
REGISTRE_CSV = EXP_DIR / "registre.csv"
SUB_DIR = USINE / "submissions"

# Paramètres du protocole (NE PAS MODIFIER en cours de projet)
N_FOLDS, SEED = 5, 42
ID, TARGET = "id", "target"


# ---------------------------------------------------------------------------
# 2. Métrique officielle
# ---------------------------------------------------------------------------
def gini(y_true, y_score) -> float:
    """Gini normalisé = 2*AUC - 1 (seul l'ordre des prédictions compte)."""
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


# ---------------------------------------------------------------------------
# 3. Chargement des données
# ---------------------------------------------------------------------------
def charger_donnees(version: str = "v0"):
    """Renvoie (train, test) en DataFrame, triés par id.

    v0 : CSV bruts d'origine, -1 = manquant laissé tel quel, aucune transformation.
    Les versions suivantes (v1, v2...) seront produites par le data engineer dans
    usine/data/processed/<version>/{train,test}.parquet.
    """
    if version == "v0":
        train, test = pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
    else:
        d = USINE / "data" / "processed" / version
        train, test = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")
    return train.sort_values(ID, ignore_index=True), test.sort_values(ID, ignore_index=True)


def liste_features(train: pd.DataFrame, exclure=()) -> list[str]:
    """Toutes les colonnes sauf id/target et les colonnes explicitement exclues."""
    return [c for c in train.columns if c not in (ID, TARGET, *exclure)]


# ---------------------------------------------------------------------------
# 4. Protocole de validation
# ---------------------------------------------------------------------------
def creer_folds(train: pd.DataFrame) -> pd.DataFrame:
    """StratifiedKFold(5, shuffle, 42) sur la cible -> DataFrame (id, fold)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(train), -1, dtype=int)
    for k, (_, idx_val) in enumerate(skf.split(train, train[TARGET])):
        fold[idx_val] = k
    return pd.DataFrame({ID: train[ID].values, "fold": fold})


def charger_folds(train: pd.DataFrame) -> np.ndarray:
    """Lit folds.csv (le crée au premier appel) et renvoie le n° de fold aligné sur train."""
    if not FOLDS_CSV.exists():
        creer_folds(train).to_csv(FOLDS_CSV, index=False)
        print(f"[folds] créé : {FOLDS_CSV}")
    folds = pd.read_csv(FOLDS_CSV)
    fold = train[[ID]].merge(folds, on=ID, how="left")["fold"]
    assert fold.notna().all() and len(fold) == len(train), "folds.csv incohérent avec le train"
    return fold.to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# 5. Registre des expériences
# ---------------------------------------------------------------------------
COLS_REGISTRE = ["id", "heure", "agent", "donnees_version", "modele", "changement", "metrique",
                 "score_cv_moyen", "score_cv_std", "seuil", "delta_vs_ref", "verdict", "commentaire"]


def ajouter_registre(ligne: dict) -> dict:
    """Ajoute une ligne au registre (fins de ligne LF).

    delta_vs_ref = différence APPARIÉE moyenne par fold vs l'expérience de référence indiquée
    (Gini_fold(exp) - Gini_fold(ref), ref nommée dans le commentaire) ; « n/a » sans référence.
    (Renommée depuis delta_vs_meilleur, qui comparait à tort au meilleur score du registre.)
    Le verdict vient de la comparaison appariée (03_train.comparer, règle 5/5 ou Δ > 2·sd/√5) ;
    sans référence -> « reference »."""
    d = ligne.get("delta_vs_ref", "n/a")
    ligne["delta_vs_ref"] = f"{d:+.5f}" if isinstance(d, (int, float)) else d
    ligne.setdefault("verdict", "reference")
    ligne.setdefault("heure", dt.datetime.now().strftime("%H:%M"))
    ligne.setdefault("agent", "usine-ml-engineer")
    ligne.setdefault("metrique", "gini")
    ligne.setdefault("seuil", "n/a")
    for k in ("score_cv_moyen", "score_cv_std"):
        ligne[k] = f"{ligne[k]:.5f}"
    with open(REGISTRE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=COLS_REGISTRE, lineterminator="\n").writerow(
            {c: ligne.get(c, "") for c in COLS_REGISTRE})
    return ligne


# ---------------------------------------------------------------------------
# 6. Soumission : écriture + contrôle strict du format
# ---------------------------------------------------------------------------
def ecrire_soumission(ids, preds, chemin) -> None:
    """CSV SANS en-tête, 2 colonnes id,prediction, même ordre que test.csv."""
    sub = pd.DataFrame({ID: np.asarray(ids, dtype=np.int64), "prediction": np.asarray(preds, dtype=float)})
    sub.to_csv(chemin, index=False, header=False, float_format="%.8f", lineterminator="\n")


def verifier_soumission(chemin, train_target_mean: float | None = None) -> bool:
    """Contrôle : nb lignes = test, pas d'en-tête, ids identiques et dans le même ordre
    que test.csv, pas de NaN, probabilités dans [0,1]. Lève AssertionError sinon."""
    ids_test = pd.read_csv(TEST_CSV, usecols=[ID])[ID].to_numpy()
    with open(chemin, encoding="utf-8") as f:
        premiere = f.readline().strip()
    assert premiere.split(",")[0].isdigit(), f"en-tête détecté : {premiere!r}"
    sub = pd.read_csv(chemin, header=None, names=[ID, "prediction"])
    assert sub.shape == (len(ids_test), 2), f"forme {sub.shape} != ({len(ids_test)}, 2)"
    assert (sub[ID].to_numpy() == ids_test).all(), "ids différents / ordre différent de test.csv"
    assert sub[ID].is_monotonic_increasing, "ids non croissants"
    assert sub["prediction"].notna().all(), "NaN dans les prédictions"
    assert sub["prediction"].between(0, 1).all(), "prédictions hors [0,1]"
    p = sub["prediction"]
    print(f"[soumission OK] {chemin} : {len(sub)} lignes, sans en-tête, ids = test.csv, "
          f"min={p.min():.5f} max={p.max():.5f} moyenne={p.mean():.5f}"
          + (f" (taux positifs train = {train_target_mean:.5f})" if train_target_mean is not None else ""))
    return True
