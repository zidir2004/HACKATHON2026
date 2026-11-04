"""02_prepare.py - Préparation des données (data cleaner). Rejouable de bout en bout.

Une transformation par version, pour mesurer chacune séparément :
  v1 = brut sans les 20 ps_calc_*                      (37 variables)
  v2 = v1 + indicateurs de manquants                   (nb_manquants + 5 *_manq)
  v3 = v2 + nettoyage des redondances                  (ps_ind_14 retirée, one-hot -> 2 catégorielles)

Règles : les -1 d'origine restent (pas d'imputation, pas d'écrêtage, aucune ligne supprimée),
aucun encodage cible ici (il se fait dans les folds, dans 03_train.py).
Sortie : usine/data/processed/<vX>/{train,test}.parquet + schema.json
Lancement (cwd = dossier Hackathon) : python -u usine/src/02_prepare.py
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
INTERIM = ROOT / "usine" / "data" / "interim"
OUT = ROOT / "usine" / "data" / "processed"
ID, TARGET = "id", "target"
MANQ_VARS = ["ps_car_03_cat", "ps_car_05_cat", "ps_reg_03", "ps_car_07_cat", "ps_ind_05_cat"]
G_IND = ["ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin"]
G_16 = ["ps_ind_16_bin", "ps_ind_17_bin", "ps_ind_18_bin"]
stats = {}   # compteurs pour le rapport


# ---------------------------------------------------------------- transformations
def retirer_calc(df):
    """v1 : retire les 20 variables ps_calc_*."""
    return df.drop(columns=[c for c in df.columns if c.startswith("ps_calc_")])


def ajouter_indicateurs_manquants(df, vars_ref):
    """v2 : nb_manquants (nombre de -1 par ligne sur les variables hors calc)
    et <var>_manq (0/1) pour les 5 variables à manquants informatifs. Les -1 restent."""
    df = df.copy()
    df["nb_manquants"] = (df[vars_ref] == -1).sum(axis=1).astype("int8")
    for v in MANQ_VARS:
        df[v + "_manq"] = (df[v] == -1).astype("int8")
    return df


def _code_onehot(df, cols, avec_aucune):
    """Code 0..k-1 = position de la colonne à 1 ; k = 'aucune' si avec_aucune."""
    s = df[cols].sum(axis=1)
    assert s.le(1).all() if avec_aucune else s.eq(1).all(), f"{cols} : ce n'est pas un one-hot"
    code = np.argmax(df[cols].to_numpy(), axis=1)
    if avec_aucune:
        code = np.where(s.to_numpy() == 0, len(cols), code)
    return code.astype("int8")


def nettoyer_redondances(df):
    """v3 : retire ps_ind_14 (= somme de ps_ind_10..13_bin) ; remplace les one-hot par 2 catégorielles."""
    df = df.copy()
    assert (df["ps_ind_14"] == df[[f"ps_ind_{i}_bin" for i in (10, 11, 12, 13)]].sum(axis=1)).all()
    c1, c2 = _code_onehot(df, G_IND, False), _code_onehot(df, G_16, True)
    # bijection : chaque combinaison de binaires <-> un seul code, et inversement
    for cols, c in ((G_IND, c1), (G_16, c2)):
        g = df[cols].astype(str).agg("".join, axis=1)
        assert pd.Series(c).groupby(g.values).nunique().eq(1).all(), "codes non uniques par combinaison"
        assert g.groupby(c).nunique().eq(1).all(), "un code pour plusieurs combinaisons"
    pos = df.columns.get_loc(G_IND[0])
    df = df.drop(columns=["ps_ind_14", *G_IND, *G_16])
    df.insert(pos, "ps_ind_06_09_cat", c1)
    df.insert(pos + 1, "ps_ind_16_18_cat", c2)
    return df


# ---------------------------------------------------------------- schéma
def construire_schema(df, version):
    feats = [c for c in df.columns if c not in (ID, TARGET)]
    cat = [c for c in feats if c.endswith("_cat")]
    binc = [c for c in feats if c.endswith("_bin") or c.endswith("_manq")]
    num = [c for c in feats if c not in cat and c not in binc]
    return {
        "version": version, "id": ID, "target": TARGET, "n_features": len(feats),
        "cat_cols": cat, "binary_cols": binc, "numeric_cols": num,
        "missing_code": -1,
        "missing_note": "Les -1 sont conservés tels quels dans les parquet. Pour LightGBM natif, les cat_cols "
                        "sont des entiers >= -1 : recoder -1 en catégorie à part au moment de l'entraînement "
                        "(ex. +1 sur la colonne, ou astype('category')). Pour les numeric_cols, -1 peut devenir "
                        "NaN (natif LightGBM) ou rester -1.",
        "cat_cols_with_missing": [c for c in cat if (df[c] == -1).any()],
        "numeric_cols_with_missing": [c for c in num if (df[c] == -1).any() and c != "nb_manquants"],
        "ordinal_like_numeric": [c for c in num if c in ("ps_ind_01", "ps_ind_03", "ps_ind_15", "ps_car_11", "nb_manquants")],
    }


# ---------------------------------------------------------------- contrôles
def controles(raw_tr, raw_te, tr, te, version):
    assert tr[ID].equals(raw_tr[ID]) and te[ID].equals(raw_te[ID]), "ids/ordre modifiés"
    assert len(tr) == len(raw_tr) and len(te) == len(raw_te)
    assert tr[TARGET].equals(raw_tr[TARGET]), "target modifiée"
    assert TARGET not in te.columns
    assert [c for c in tr.columns if c != TARGET] == list(te.columns), "colonnes train != test"
    assert not tr.isna().any().any() and not te.isna().any().any(), "NaN introduit"
    assert tr[ID].is_unique and te[ID].is_unique
    for c in tr.columns:   # colonnes conservées = identiques à l'origine
        if c in raw_tr.columns:
            assert tr[c].equals(raw_tr[c]), f"{c} modifiée"
    print(f"[controles OK] {version} : train {tr.shape}, test {te.shape}")


def ecrire(tr, te, version, raw_tr, raw_te):
    controles(raw_tr, raw_te, tr, te, version)
    d = OUT / version
    d.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(d / "train.parquet", index=False)
    te.to_parquet(d / "test.parquet", index=False)
    schema = construire_schema(tr, version)
    (d / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    stats[version] = {"cols": tr.shape[1] - 2, "cat": len(schema["cat_cols"]), "bin": len(schema["binary_cols"]),
                      "num": len(schema["numeric_cols"])}
    print(version, stats[version])


def main():
    raw_tr, raw_te = pd.read_parquet(INTERIM / "train_raw.parquet"), pd.read_parquet(INTERIM / "test_raw.parquet")
    v1 = [retirer_calc(d) for d in (raw_tr, raw_te)]
    ref = [c for c in v1[1].columns if c != ID]   # variables hors calc
    ecrire(*v1, "v1", raw_tr, raw_te)
    v2 = [ajouter_indicateurs_manquants(d, ref) for d in v1]
    ecrire(*v2, "v2", raw_tr, raw_te)
    v3 = [nettoyer_redondances(d) for d in v2]
    ecrire(*v3, "v3", raw_tr, raw_te)
    tr2, tr3 = v2[0], v3[0]
    out = {"n_train": len(tr2), "n_test": len(v2[1]),
           "nb_manquants_dist": tr2["nb_manquants"].value_counts().sort_index().to_dict(),
           "manq_taux": {v: float(tr2[v + "_manq"].mean()) for v in MANQ_VARS},
           "manq_target_rate": {v: [float(tr2.loc[tr2[v + "_manq"] == 1, TARGET].mean()),
                                    float(tr2.loc[tr2[v + "_manq"] == 0, TARGET].mean())] for v in MANQ_VARS},
           "ps_ind_06_09_cat": tr3["ps_ind_06_09_cat"].value_counts().sort_index().to_dict(),
           "ps_ind_16_18_cat": tr3["ps_ind_16_18_cat"].value_counts().sort_index().to_dict(),
           "stats": stats}
    (OUT / "prep_stats.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
