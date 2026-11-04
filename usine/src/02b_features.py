"""02b_features.py - Feature engineering (data cleaner) : v4 = v1 + variables candidates.

Une fonction par famille. Aucune utilisation de la cible.
  F1 comptages (frequency encoding) calculés sur train+test concaténés (statistique NON supervisée,
     donc sans fuite ; la distribution du test est utilisée, c'est légitime en hackathon)
  F2 interactions numériques (ligne par ligne)
  F3 entiers reconstruits
  F4 somme des binaires ps_ind_*_bin
  F5 croisement de 2 catégorielles puis comptage
Les -1 d'origine sont conservés (-1 propagé quand une opérande est manquante).
Sortie : usine/data/processed/v4/{train,test}.parquet + schema.json ; usine/data/processed/v4_gini_univarie.csv
Lancement (cwd = Hackathon) : python -u usine/src/02b_features.py
"""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "usine" / "data" / "processed"
ID, TARGET = "id", "target"

spec = importlib.util.spec_from_file_location("prep", ROOT / "usine" / "src" / "02_prepare.py")
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)

NEW = {}   # nom -> famille


def f1_comptages(tr, te):
    """F1 : fréquence de chaque modalité sur train+test (-1 compte comme une modalité)."""
    allx = pd.concat([tr, te], ignore_index=True)
    for c in ["ps_car_11_cat", "ps_car_01_cat", "ps_car_06_cat"]:
        vc = allx[c].value_counts()
        for d in (tr, te):
            d[c + "_count"] = d[c].map(vc).astype("int32")
        NEW[c + "_count"] = "F1"


def f2_interactions(d):
    """F2 : produits/ratios. ps_reg_03 = -1 -> -1 ; ps_car_12 <= 0 (un -1 dans le train) -> -1."""
    d["reg01_x_reg02"] = d["ps_reg_01"] * d["ps_reg_02"]
    d["car13_x_reg03"] = np.where(d["ps_reg_03"] == -1, -1.0, d["ps_car_13"] * d["ps_reg_03"])
    ok = d["ps_car_12"] > 0
    d["car13_div_car12"] = np.where(ok, d["ps_car_13"] / d["ps_car_12"].where(ok, 1.0), -1.0)
    d["ind03_plus_ind15"] = d["ps_ind_03"] + d["ps_ind_15"]
    d["ind03_x_ind15"] = d["ps_ind_03"] * d["ps_ind_15"]
    for c in ["reg01_x_reg02", "car13_x_reg03", "car13_div_car12", "ind03_plus_ind15", "ind03_x_ind15"]:
        NEW[c] = "F2"


def f3_entiers(d):
    """F3 : ps_car_15 = racine d'un entier ; ps_reg_03 = racine d'(entier/1600) = racine(entier)/40."""
    d["ps_car_15_int"] = np.rint(d["ps_car_15"] ** 2).astype("int16")
    d["ps_reg_03_int"] = np.where(d["ps_reg_03"] == -1, -1, np.rint((d["ps_reg_03"] * 40) ** 2)).astype("int32")
    NEW["ps_car_15_int"] = NEW["ps_reg_03_int"] = "F3"


def f4_somme_bin(d):
    """F4 : nombre de binaires ps_ind_*_bin à 1."""
    cols = [c for c in d.columns if c.startswith("ps_ind_") and c.endswith("_bin")]
    d["ps_ind_bin_sum"] = d[cols].sum(axis=1).astype("int8")
    NEW["ps_ind_bin_sum"] = "F4"


def f5_croisement(tr, te):
    """F5 : ps_ind_05_cat x ps_car_04_cat -> chaîne -> comptage sur train+test."""
    key = lambda d: d["ps_ind_05_cat"].astype(str) + "_" + d["ps_car_04_cat"].astype(str)
    vc = pd.concat([key(tr), key(te)]).value_counts()
    for d in (tr, te):
        d["ind05_x_car04_count"] = key(d).map(vc).astype("int32")
    NEW["ind05_x_car04_count"] = "F5"


def gini_univarie(tr, cols):
    """Gini = 2*AUC-1 de la variable brute comme score, par fold (folds.csv), moyenne des folds.
    Signe libre : on rapporte la moyenne signée et sa valeur absolue."""
    folds = pd.read_csv(ROOT / "usine" / "experiences" / "folds.csv").set_index(ID)["fold"]
    f = folds.reindex(tr[ID]).to_numpy()
    y = tr[TARGET].to_numpy()
    rows = []
    for c in cols:
        x = tr[c].to_numpy(dtype=float)
        g = []
        for k in range(5):
            m = f == k
            r = pd.Series(x[m]).rank().to_numpy()
            n1 = y[m].sum(); n0 = m.sum() - n1
            auc = (r[y[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            g.append(2 * auc - 1)
        rows.append({"variable": c, "famille": NEW.get(c, "ref"), "gini_moy": np.mean(g),
                     "gini_abs": abs(np.mean(g)), "std_folds": np.std(g), "n_distinct": tr[c].nunique()})
    return pd.DataFrame(rows)


def controles(tr, te, v1tr, v1te):
    assert tr[ID].equals(v1tr[ID]) and te[ID].equals(v1te[ID]) and tr[TARGET].equals(v1tr[TARGET])
    assert [c for c in tr.columns if c != TARGET] == list(te.columns), "colonnes train != test"
    assert not tr.isna().any().any() and not te.isna().any().any(), "NaN"
    assert not np.isinf(tr.select_dtypes("number").to_numpy(dtype=float)).any(), "inf"
    for c in v1tr.columns:
        assert tr[c].equals(v1tr[c]), f"{c} modifiée"
    assert ((tr["car13_x_reg03"] == -1) == (tr["ps_reg_03"] == -1)).all()
    assert ((tr["ps_reg_03_int"] == -1) == (tr["ps_reg_03"] == -1)).all()
    # -1 cohérents pour le ratio
    assert ((tr["car13_div_car12"] == -1) == (tr["ps_car_12"] <= 0)).all()
    # entiers reconstruits : erreur d'arrondi petite
    assert np.abs(np.sqrt(tr["ps_car_15_int"]) - tr["ps_car_15"]).max() < 1e-4
    r = tr["ps_reg_03"] != -1
    print("max err reg03_int:", np.abs(np.sqrt(tr.loc[r, "ps_reg_03_int"]) / 40 - tr.loc[r, "ps_reg_03"]).max())
    print("[controles OK] v4", tr.shape, te.shape)


def main():
    v1tr, v1te = pd.read_parquet(PROC / "v1" / "train.parquet"), pd.read_parquet(PROC / "v1" / "test.parquet")
    tr, te = v1tr.copy(), v1te.copy()
    f1_comptages(tr, te); f5_croisement(tr, te)
    for d in (tr, te):
        f2_interactions(d); f3_entiers(d); f4_somme_bin(d)
    controles(tr, te, v1tr, v1te)
    d = PROC / "v4"; d.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(d / "train.parquet", index=False); te.to_parquet(d / "test.parquet", index=False)
    schema = prep.construire_schema(tr, "v4")
    schema["new_features"] = NEW
    schema["note"] = "v4 = v1 + variables candidates (famille F1..F5). Comptages calculés sur train+test (non supervisé)."
    (d / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    ref = ["ps_car_11_cat", "ps_car_01_cat", "ps_car_06_cat", "ps_reg_01", "ps_reg_02", "ps_reg_03", "ps_car_13",
           "ps_car_12", "ps_ind_03", "ps_ind_15", "ps_car_15", "ps_ind_05_cat", "ps_car_04_cat"]
    g = gini_univarie(tr, list(NEW) + ref)
    g.to_csv(PROC / "v4_gini_univarie.csv", index=False)
    print(g.round(4).to_string())
    print(json.dumps({k: schema[k] for k in ("n_features", "cat_cols", "binary_cols")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
