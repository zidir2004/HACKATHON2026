"""03_train.py — entraînement en validation croisée (protocole unique du projet).

Usage (depuis le dossier Hackathon) :
    python -u usine/src/03_train.py --exp E02 E03      # enchaîne les expériences

Chaque expérience est décrite par une CONFIG (section 1) : version de données, features
exclues, modèle, paramètres, options de préparation (encodage cible, catégorielles natives),
expérience de RÉFÉRENCE (ref) et changement testé. Une expérience = UN changement vs ref.

Étapes pour chaque expérience :
  1. Chargement des données (version) + folds communs (usine/experiences/folds.csv)
  2. Boucle sur les 5 folds : préparation apprise DANS le fold d'entraînement (encodage cible
     avec K-fold interne), ajustement, early stopping sur le fold de validation,
     prédiction OOF du fold + prédiction du test
  3. Gini par fold (moyenne ± écart-type), Gini OOF global
  4. Comparaison APPARIÉE par fold vs la référence (Gini_fold(exp) - Gini_fold(ref)),
     moyenne, nb de folds gagnés -> verdict (règle du reviewer)
  5. Sauvegardes : oof/<EXP>_oof.csv, oof/<EXP>_test.csv, oof/<EXP>_folds_detail.csv,
     comparaisons.csv, ligne dans registre.csv

Métrique : Gini = 2*AUC - 1 -> aucun seuil à optimiser (seuil = n/a).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import utils as U  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Configurations des expériences
# ---------------------------------------------------------------------------
LGB_BASE = dict(
    objective="binary", metric="auc", learning_rate=0.02, num_leaves=24,
    min_child_samples=150, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=5.0, max_bin=255, n_jobs=16, seed=U.SEED, verbose=-1,
)
LGB_CAT = dict(LGB_BASE, cat_smooth=50, min_data_per_group=200, max_cat_threshold=32, cat_l2=10)
XGB_BASE = dict(
    objective="binary:logistic", eval_metric="auc", device="cuda", tree_method="hist",
    eta=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
    reg_lambda=5.0, reg_alpha=1.0, max_bin=256, seed=U.SEED,
)
CB_BASE = dict(
    loss_function="Logloss", eval_metric="AUC", task_type="GPU", depth=6, learning_rate=0.05,
    l2_leaf_reg=5, border_count=254, random_seed=U.SEED, verbose=0,
)
LGB = dict(modele="lightgbm", params=LGB_BASE, n_max=3000, early_stop=200)

CONFIGS = {
    "E00": dict(version="v0", modele="logreg", params=dict(C=0.1, max_iter=1000),
                changement="repère naïf : régression logistique L2 sur variables brutes standardisées"),
    "E01": dict(version="v0", **LGB,
                changement="baseline LightGBM sur données brutes v0 (-1 laissé tel quel)"),
    # --- Données (un changement à la fois, LightGBM E01 inchangé) ---
    "E02": dict(version="v1", ref="E01", **LGB, changement="v1 : retrait des 20 ps_calc_*"),
    "E03": dict(version="v2", ref="E02", **LGB,
                changement="v2 : + nb_manquants et 5 indicateurs *_manq"),
    "E04": dict(version="v3", ref="E03", **LGB,
                changement="v3 : sans ps_ind_14, one-hot regroupés (ps_ind_06_09_cat, ps_ind_16_18_cat)"),
    # --- Réglage léger du LightGBM retenu (E02, v1) : un hyperparamètre à la fois ---
    "E10": dict(version="v1", ref="E02", **dict(LGB, params=dict(LGB_BASE, feature_fraction=0.4)),
                changement="LightGBM v1 : feature_fraction 0.6 -> 0.4"),
    "E11": dict(version="v1", ref="E02", **dict(LGB, params=dict(LGB_BASE, num_leaves=16)),
                changement="LightGBM v1 : num_leaves 24 -> 16"),
    "E12": dict(version="v1", ref="E02", **dict(LGB, params=dict(LGB_BASE, min_child_samples=500)),
                changement="LightGBM v1 : min_child_samples 150 -> 500"),
    "E13": dict(version="v1", ref="E02", modele="lightgbm", params=dict(LGB_BASE, learning_rate=0.01),
                n_max=6000, early_stop=400, changement="LightGBM v1 : learning_rate 0.02 -> 0.01 (patience 400)"),
}


# --- v4 : feature engineering (data cleaner, rapport 07). Ablation par famille via la liste des
#     colonnes de v4 : v4 = v1 + 12 candidates ; on exclut les candidates non testées. Config E02. ---
V4_NOUVELLES = {
    "F1": ["ps_car_11_cat_count", "ps_car_01_cat_count", "ps_car_06_cat_count"],
    "F5": ["ind05_x_car04_count"],
    "F2": ["car13_div_car12", "car13_x_reg03"],          # les 2 ratios/produits car13 (prioritaires)
    "F2b": ["reg01_x_reg02", "ind03_plus_ind15", "ind03_x_ind15"],
    "F3": ["ps_car_15_int", "ps_reg_03_int"],
    "F4": ["ps_ind_bin_sum"],
}
TOUTES_V4 = [c for f in V4_NOUVELLES.values() for c in f]


def v4_avec(familles):
    """Exclusions pour garder v1 + les familles données (sélection de colonnes de v4)."""
    garder = {c for f in familles for c in V4_NOUVELLES[f]}
    return [c for c in TOUTES_V4 if c not in garder]


CONFIGS.update({
    "E19": dict(version="v4", ref="E02", **LGB, changement="v4 complète : v1 + 12 variables candidates (F1-F5)"),
    "E20": dict(version="v4", ref="E02", exclure=v4_avec(["F5"]), **LGB,
                changement="v1 + F5 (ind05_x_car04_count, comptage du couple)"),
    "E21": dict(version="v4", ref="E02", exclure=v4_avec(["F2"]), **LGB,
                changement="v1 + F2 (car13_div_car12, car13_x_reg03)"),
    "E22": dict(version="v4", ref="E02", exclure=v4_avec(["F1"]), **LGB,
                changement="v1 + F1 (comptages ps_car_11/01/06_cat)"),
    # Fin d'ablation (familles « en dernier » du data cleaner), pour expliquer le recul de E19
    "E25": dict(version="v4", ref="E02", exclure=v4_avec(["F2b"]), **LGB,
                changement="v1 + F2b (reg01_x_reg02, ind03_plus_ind15, ind03_x_ind15)"),
    "E26": dict(version="v4", ref="E02", exclure=v4_avec(["F3"]), **LGB,
                changement="v1 + F3 (ps_car_15_int, ps_reg_03_int)"),
    "E27": dict(version="v4", ref="E02", exclure=v4_avec(["F4"]), **LGB,
                changement="v1 + F4 (ps_ind_bin_sum)"),
})


# --- Piste A : LightGBM de la solution 1re place Kaggle Porto Seguro (M. Jahrer), paramètres publiés.
#     Nombre d'arbres FIXE (1400, pas d'early stopping -> pas de biais optimiste sur le fold rapporté). ---
LGB_JAHRER = dict(
    objective="binary", metric="auc", boosting="gbdt", learning_rate=0.01, max_bin=255, num_leaves=31,
    min_data_in_leaf=1500, feature_fraction=0.7, bagging_freq=1, bagging_fraction=0.7,
    lambda_l1=1.0, lambda_l2=1.0, n_jobs=8, seed=U.SEED, verbose=-1,
)
LGB_J = dict(modele="lightgbm", params=LGB_JAHRER, n_max=1400, early_stop=None)
CONFIGS.update({
    "E29": dict(version="v1", ref="E13", **LGB_J,
                changement="LightGBM paramètres Jahrer (1re place Porto Seguro), 1400 arbres fixes, _cat numériques"),
    "E30": dict(version="v1", ref="E29", **LGB_J, onehot="remplace",
                changement="E29 + one-hot des 14 _cat (-1 = modalité à part), colonnes _cat d'origine RETIRÉES"),
    "E30b": dict(version="v1", ref="E29", **LGB_J, onehot="ajoute",
                 changement="E29 + one-hot des 14 _cat (-1 = modalité à part), colonnes _cat d'origine GARDÉES"),
    # 1400 arbres réglés par Jahrer pour 476k lignes d'entraînement/fold ; nous en avons 357k -> mise à
    # l'échelle A PRIORI (non choisie sur la validation) : 1400 x 357/476 ≈ 1050.
    "E35": dict(version="v1", ref="E30b", **dict(LGB_J, n_max=1050), onehot="ajoute",
                changement="E30b avec 1050 arbres au lieu de 1400 (mise à l'échelle a priori sur la taille du train)"),
})


def ajouter_configs_v4(familles, exp_lgb="E23", exp_xgb="E24x", exp_cb="E24c"):
    """Union de familles retenues : LightGBM (ref E02) + XGBoost/CatBoost GPU (ref E07/E08)."""
    ex, lib = v4_avec(familles), "+".join(familles)
    CONFIGS.update({
        exp_lgb: dict(version="v4", ref="E02", exclure=ex, **LGB, changement=f"v1 + {lib}"),
        exp_xgb: dict(version="v4", ref="E07", exclure=ex, modele="xgboost", params=XGB_BASE, n_max=6000,
                      early_stop=200, changement=f"XGBoost GPU (config E07) sur v1 + {lib}"),
        exp_cb: dict(version="v4", ref="E08", exclure=ex, modele="catboost", params=CB_BASE, n_max=6000,
                     early_stop=200, changement=f"CatBoost GPU (config E08) sur v1 + {lib}"),
    })


def ajouter_configs_dynamiques(base_exp: str, base_version: str, te_cols=()):
    """Expériences E05+ définies à partir de la meilleure retenue (passée en argument)."""
    b = dict(version=base_version, ref=base_exp)
    CONFIGS.update({
        "E05": dict(b, **LGB, te_cols=["ps_car_11_cat"],
                    changement="+ encodage cible lissé de ps_car_11_cat (dans les folds, K-fold interne)"),
        "E06": dict(b, **dict(LGB, params=LGB_CAT), cat_native=True,
                    changement="toutes les _cat en catégorielles natives LightGBM (-1 = modalité à part)"),
        # Modèles comparatifs sur la meilleure version de données (ref = meilleur LightGBM)
        "E07": dict(b, modele="xgboost", params=XGB_BASE, n_max=6000, early_stop=200, te_cols=te_cols,
                    changement="XGBoost GPU (depth 5, eta 0.03, subsample 0.8, colsample 0.6, mcw 10)"),
        "E15": dict(b, modele="catboost", params=dict(CB_BASE, learning_rate=0.03), n_max=8000,
                    early_stop=300, ref="E08", changement="CatBoost GPU : lr 0.05 -> 0.03 (patience 300)"),
        "E16": dict(b, modele="xgboost", params=dict(XGB_BASE, eta=0.015), n_max=10000, early_stop=400,
                    te_cols=te_cols, ref="E07", changement="XGBoost GPU : eta 0.03 -> 0.015 (patience 400)"),
        "E08": dict(b, modele="catboost", params=CB_BASE, n_max=6000, early_stop=200,
                    changement="CatBoost GPU (depth 6, lr 0.05, _cat en cat_features natives)"),
    })


CLE_GRAINE = {"lightgbm": "seed", "xgboost": "seed", "catboost": "random_seed"}


def ajouter_configs_graines(membres=("E13", "E07", "E08"), graines=(7, 2026)):
    """Variantes « graine de modèle » d'expériences existantes : <EXP>s<graine>.
    Seule la graine du modèle change (mêmes folds, mêmes données, mêmes paramètres).
    Ce sont des ingrédients de la moyenne multi-graines (E17) : pas de ligne au registre."""
    for m in membres:
        for g in graines:
            c = dict(CONFIGS[m])
            c["params"] = dict(c["params"], **{CLE_GRAINE[c["modele"]]: g})
            c.update(ref=m, sans_registre=True, changement=f"{m} avec graine de modèle {g} (au lieu de 42)")
            CONFIGS[f"{m}s{g}"] = c


# ---------------------------------------------------------------------------
# 2. Préparations apprises dans le fold
# ---------------------------------------------------------------------------
def encodage_cible(x_tr, y_tr, x_autres, m=100.0, k_interne=5):
    """Moyenne de la cible lissée : (n*moy + m*prior)/(n+m).
    Lignes d'entraînement : K-fold interne (chaque ligne encodée sans sa propre cible).
    Autres lignes (validation, test) : encodage appris sur tout le fold d'entraînement."""
    prior = y_tr.mean()

    def table(x, y):
        s = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["sum", "count"])
        return (s["sum"] + m * prior) / (s["count"] + m)

    enc_tr = np.full(len(x_tr), prior)
    skf = StratifiedKFold(n_splits=k_interne, shuffle=True, random_state=U.SEED)
    for a, b in skf.split(x_tr, y_tr):
        t = table(x_tr[a], y_tr[a])
        enc_tr[b] = pd.Series(x_tr[b]).map(t).fillna(prior).to_numpy()
    t = table(x_tr, y_tr)
    return enc_tr, [pd.Series(x).map(t).fillna(prior).to_numpy() for x in x_autres]


def one_hot(X, Xte, cats, garder_origine=False):
    """One-hot des colonnes catégorielles (-1 = modalité à part). NON supervisé (aucune cible) :
    les modalités sont celles de train ∪ test, colonne par colonne -> pas besoin de le faire dans les folds."""
    nouv, nouv_te = [], []
    for c in cats:
        mods = np.union1d(X[c].unique(), Xte[c].unique())
        for m in mods:
            nom = f"{c}_oh{int(m)}"
            nouv.append((X[c] == m).astype(np.float32).rename(nom))
            nouv_te.append((Xte[c] == m).astype(np.float32).rename(nom))
    garde = list(X.columns) if garder_origine else [c for c in X.columns if c not in cats]
    X2 = pd.concat([X[garde], *nouv], axis=1)
    Xte2 = pd.concat([Xte[garde], *nouv_te], axis=1)
    print(f"  one-hot : {len(nouv)} colonnes indicatrices ajoutées, origine {'gardée' if garder_origine else 'retirée'}"
          f" -> {X2.shape[1]} variables", flush=True)
    return X2, Xte2, (cats if garder_origine else [])


def colonnes_cat(version: str, feats: list[str]) -> list[str]:
    if version == "v0":
        return [c for c in feats if c.endswith("_cat")]
    schema = json.loads((U.USINE / "data" / "processed" / version / "schema.json").read_text(encoding="utf-8"))
    return [c for c in schema["cat_cols"] if c in feats]


# ---------------------------------------------------------------------------
# 3. Modèles : (cfg, Xtr, ytr, Xva, yva, Xte, cats) -> (pred_val, pred_test, info)
# ---------------------------------------------------------------------------
def ajuster_lightgbm(cfg, Xtr, ytr, Xva, yva, Xte, cats):
    cf = cats if cfg.get("cat_native") else "auto"
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cf, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cf)
    if cfg.get("early_stop") is None:
        # Nombre d'arbres FIXE : la courbe de validation n'est que lue (diagnostic), jamais utilisée
        # pour choisir l'itération -> le Gini du fold n'a pas de biais optimiste.
        hist = {}
        model = lgb.train(cfg["params"], dtr, num_boost_round=cfg["n_max"], valid_sets=[dva],
                          callbacks=[lgb.record_evaluation(hist)])
        auc = hist["valid_0"]["auc"]
        return (model.predict(Xva), model.predict(Xte),
                {"n_arbres": cfg["n_max"], "diag_iter_max_auc": int(np.argmax(auc)) + 1})
    model = lgb.train(cfg["params"], dtr, num_boost_round=cfg["n_max"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg["early_stop"], verbose=False)])
    it = model.best_iteration
    return model.predict(Xva, num_iteration=it), model.predict(Xte, num_iteration=it), {"n_arbres": it}


def ajuster_xgboost(cfg, Xtr, ytr, Xva, yva, Xte, cats):
    import xgboost as xgb
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=cfg["params"]["max_bin"])
    dva = xgb.DMatrix(Xva, yva)
    model = xgb.train(cfg["params"], dtr, num_boost_round=cfg["n_max"], evals=[(dva, "val")],
                      early_stopping_rounds=cfg["early_stop"], verbose_eval=False)
    r = (0, model.best_iteration + 1)
    return (model.predict(dva, iteration_range=r), model.predict(xgb.DMatrix(Xte), iteration_range=r),
            {"n_arbres": model.best_iteration + 1})


def ajuster_catboost(cfg, Xtr, ytr, Xva, yva, Xte, cats):
    from catboost import CatBoostClassifier, Pool
    def pool(X, y=None):
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(np.int64)
        return Pool(X, y, cat_features=cats)
    model = CatBoostClassifier(**cfg["params"], iterations=cfg["n_max"], od_type="Iter",
                               od_wait=cfg["early_stop"], use_best_model=True)
    model.fit(pool(Xtr, ytr), eval_set=pool(Xva, yva))
    return (model.predict_proba(pool(Xva))[:, 1], model.predict_proba(pool(Xte))[:, 1],
            {"n_arbres": model.get_best_iteration() + 1})


def ajuster_logreg(cfg, Xtr, ytr, Xva, yva, Xte, cats):
    model = make_pipeline(StandardScaler(), LogisticRegression(**cfg["params"]))
    model.fit(Xtr, ytr)
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1], {"n_arbres": "n/a"}


MODELES = {"lightgbm": ajuster_lightgbm, "xgboost": ajuster_xgboost,
           "catboost": ajuster_catboost, "logreg": ajuster_logreg}


# ---------------------------------------------------------------------------
# 4. Comparaison appariée par fold + verdict (règle du reviewer)
# ---------------------------------------------------------------------------
def comparer(exp, ref, ginis, enregistrer=True):
    """Différence appariée Gini_fold(exp) - Gini_fold(ref) sur les MÊMES folds.

    Règle de verdict (durcie, revue reviewer 91) :
      garde  si Δmoy > 0 ET (5/5 folds gagnés OU Δmoy > 2·sd(Δ)/√5)
      rejete si Δmoy < 0 ET (0/5 folds gagnés OU -Δmoy > 2·sd(Δ)/√5)
      neutre sinon.   sd(Δ) = écart-type des Δ par fold avec ddof=1 (erreur-type ≈ test t).
    La colonne d_std reste l'écart-type ddof=0 (inchangée pour l'historique)."""
    g_ref = pd.read_csv(U.OOF_DIR / f"{ref}_folds_detail.csv").sort_values("fold")["gini"].to_numpy()
    d = np.asarray(ginis) - g_ref
    moy, gagnes = float(d.mean()), int((d > 0).sum())
    seuil = 2.0 * float(np.std(d, ddof=1)) / np.sqrt(len(d))
    if moy > 0 and (gagnes == len(d) or moy > seuil):
        verdict = "garde"
    elif moy < 0 and (gagnes == 0 or -moy > seuil):
        verdict = "rejete"
    else:
        verdict = "neutre"
    ligne = dict(exp=exp, ref=ref, **{f"d_fold{k}": round(float(v), 5) for k, v in enumerate(d)},
                 d_moy=round(moy, 5), d_std=round(float(d.std()), 5), folds_gagnes=f"{gagnes}/5",
                 seuil_2se=round(seuil, 5), verdict=verdict)
    if not enregistrer:
        return ligne
    f = U.EXP_DIR / "comparaisons.csv"
    pd.DataFrame([{k: v for k, v in ligne.items() if k != "seuil_2se"}]).to_csv(f, mode="a", header=not f.exists(), index=False, lineterminator="\n")
    return ligne


# ---------------------------------------------------------------------------
# 5. Validation croisée
# ---------------------------------------------------------------------------
def validation_croisee(exp: str) -> dict:
    cfg = CONFIGS[exp]
    print(f"\n===== {exp} | {cfg['modele']} | données {cfg['version']} | ref {cfg.get('ref')} =====\n"
          f"{cfg['changement']}", flush=True)
    train, test = U.charger_donnees(cfg["version"])
    feats = U.liste_features(train, cfg.get("exclure", []))
    cats = colonnes_cat(cfg["version"], feats)
    X, y, Xte = train[feats].astype(np.float32), train[U.TARGET].to_numpy(), test[feats].astype(np.float32)
    if cfg.get("cat_native") or cfg["modele"] == "catboost":
        for D in (X, Xte):
            D[cats] = D[cats] + 1          # -1 (manquant) -> 0 = modalité à part, valeurs >= 0
    if cfg.get("onehot"):
        X, Xte, cats = one_hot(X, Xte, cats, garder_origine=(cfg["onehot"] == "ajoute"))
    fold = U.charger_folds(train)
    te_cols = cfg.get("te_cols", [])
    print(f"train {X.shape}, test {Xte.shape}, {len(feats)} features (+{len(te_cols)} TE), "
          f"{len(cats)} cat, taux positifs {y.mean():.4%}", flush=True)

    oof, pred_test, ginis, infos = np.zeros(len(X)), np.zeros(len(Xte)), [], []
    t0 = time.time()
    for k in range(U.N_FOLDS):
        tr, va = fold != k, fold == k
        Xtr, Xva, Xtek = X[tr].copy(), X[va].copy(), Xte.copy()
        for c in te_cols:                   # encodage cible appris dans le fold d'entraînement
            e_tr, (e_va, e_te) = encodage_cible(X.loc[tr, c].to_numpy(), y[tr],
                                                [X.loc[va, c].to_numpy(), Xte[c].to_numpy()])
            Xtr[c + "_te"], Xva[c + "_te"], Xtek[c + "_te"] = e_tr, e_va, e_te
        t = time.time()
        p_va, p_te, info = MODELES[cfg["modele"]](cfg, Xtr, y[tr], Xva, y[va], Xtek, cats)
        oof[va] = p_va
        pred_test += p_te / U.N_FOLDS
        g = U.gini(y[va], p_va)
        ginis.append(g); info.update(fold=k, gini=g, duree_s=round(time.time() - t, 1)); infos.append(info)
        print(f"  fold {k} : gini={g:.5f}  arbres={info['n_arbres']}  durée={info['duree_s']}s", flush=True)

    res = dict(exp=exp, gini_moy=float(np.mean(ginis)), gini_std=float(np.std(ginis)),
               gini_oof=U.gini(y, oof), duree_s=round(time.time() - t0, 1), folds=infos)
    print(f"  => gini CV = {res['gini_moy']:.5f} ± {res['gini_std']:.5f} | gini OOF = {res['gini_oof']:.5f}"
          f" | durée totale {res['duree_s']}s")

    U.OOF_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({U.ID: train[U.ID], U.TARGET: y, "oof": oof, "fold": fold}).to_csv(
        U.OOF_DIR / f"{exp}_oof.csv", index=False)
    pd.DataFrame({U.ID: test[U.ID], "prediction": pred_test}).to_csv(U.OOF_DIR / f"{exp}_test.csv", index=False)
    pd.DataFrame(infos).to_csv(U.OOF_DIR / f"{exp}_folds_detail.csv", index=False)

    arbres = [i["n_arbres"] for i in infos]
    comm = f"gini OOF {res['gini_oof']:.5f}; arbres/fold {arbres}; {res['duree_s']}s"
    extra = {}
    if cfg.get("sans_registre"):
        c = comparer(exp, cfg["ref"], ginis, enregistrer=False)
        print(f"  [ingrédient, hors registre] vs {c['ref']} : Δmoy={c['d_moy']:+.5f} Δ/fold "
              f"{[c[f'd_fold{k}'] for k in range(5)]}", flush=True)
        return res
    if cfg.get("ref"):
        c = comparer(exp, cfg["ref"], ginis)
        comm += (f"; apparié vs {c['ref']}: Δmoy {c['d_moy']:+.5f} (std {c['d_std']:.5f}), "
                 f"folds gagnés {c['folds_gagnes']}, Δ/fold {[c[f'd_fold{k}'] for k in range(5)]}")
        extra["verdict"], extra["delta_vs_ref"] = c["verdict"], c["d_moy"]
        print(f"  apparié vs {c['ref']} : Δmoy={c['d_moy']:+.5f} folds gagnés {c['folds_gagnes']} "
              f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} -> {c['verdict']}")
    ligne = U.ajouter_registre(dict(
        id=exp, donnees_version=cfg["version"], modele=cfg["modele"], changement=cfg["changement"],
        score_cv_moyen=res["gini_moy"], score_cv_std=res["gini_std"], commentaire=comm, **extra))
    print(f"  registre : delta_vs_ref={ligne['delta_vs_ref']} verdict={ligne['verdict']}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--base", default=None, help="expérience de base pour E05+ (ex. E04)")
    ap.add_argument("--base-version", default=None, help="version de données de la base (ex. v3)")
    ap.add_argument("--te", nargs="*", default=[], help="colonnes encodées cible pour XGBoost (si E05 gardée)")
    ap.add_argument("--graines", nargs="*", type=int, default=None,
                    help="crée les variantes <EXP>s<graine> de E13/E07/E08 (ingrédients E17)")
    ap.add_argument("--familles-v4", nargs="*", default=None,
                    help="familles v4 retenues (ex. F5 F2) -> configs E23 (LGB), E24x (XGB), E24c (CB)")
    ap.add_argument("--membres-graines", nargs="*", default=["E13", "E07", "E08"],
                    help="expériences dont on crée les variantes de graine (défaut E13 E07 E08)")
    a = ap.parse_args()
    if a.base:
        ajouter_configs_dynamiques(a.base, a.base_version, list(a.te))
    if a.familles_v4:
        ajouter_configs_v4(a.familles_v4)
    if a.graines:
        ajouter_configs_graines(membres=tuple(a.membres_graines), graines=a.graines)
    for e in a.exp:
        validation_croisee(e)
