"""17_refit_complet.py — réentraînement des 4 membres de 2_306 (E33f) sur 100 % du train.

Pourquoi : dans 2_306, la prédiction test de chaque membre est la moyenne de 5 modèles de fold,
chacun entraîné sur 80 % du train (357k lignes). Ici on réentraîne chaque membre UNE fois sur
100 % du train (446k lignes) avec exactement la même préparation et les mêmes paramètres.
Aucune CV possible : le contrôle se fait par Spearman vs la prédiction test « par folds ».

Expériences (une par appel) :
  E80 = LightGBM E30 (Jahrer + one-hot), 1680 arbres fixes = 1400 x 1,2 (train 1,25x plus grand)
  E81 = XGBoost GPU E07, arbres = moyenne des best_iteration des 5 folds de E07 x 1,2
  E82 = CatBoost GPU E08, arbres = moyenne des best_iteration des 5 folds de E08 x 1,2
  E85 = comme E80 mais 1260 arbres = 1050 x 1,2 (1050 = config E35 ; 1400 dépasse l'optimum, revue 94)
  E83 = réseau 4500-1000-1000 sur activations du DAE gelé (dae.pt, NON réentraîné),
        6 époques fixes, moyenne des prédictions test des 6 époques, 3 graines (0, 1, 2) moyennées
Réutilise : 03_train.py (configs, one_hot, colonnes_cat), 13_nn_dae.py (charger, Reseau,
representation, predire), utils.py.

Commandes (cwd = Hackathon) :
  python -u usine/src/17_refit_complet.py --exp E80          (CPU)
  python -u usine/src/17_refit_complet.py --exp E81          (GPU)
  python -u usine/src/17_refit_complet.py --exp E82          (GPU)
  python -u usine/src/17_refit_complet.py --exp E83          (GPU)
Sortie : usine/experiences/oof/<EXP>_test.csv (id,prediction) + Spearman vs l'équivalent par folds.
Le mélange E84 / la soumission 2_308 sont faits par 18_combiner.py.
"""
import argparse
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

FACTEUR = 1.2          # mise à l'échelle du nombre d'arbres : 357k -> 446k lignes
EQUIVALENT = {"E80": "E30", "E85": "E30", "E81": "E07", "E82": "E08", "E83": "E32c"}   # prédiction test par folds


# ---------------------------------------------------------------------------
# 1. Arbres : même préparation que 03_train.validation_croisee, sans boucle de folds
# ---------------------------------------------------------------------------
def n_arbres_cv(exp_cv: str) -> int:
    """Moyenne des best_iteration des 5 folds (folds_detail.csv) x FACTEUR."""
    d = pd.read_csv(U.OOF_DIR / f"{exp_cv}_folds_detail.csv")
    return int(round(d["n_arbres"].mean() * FACTEUR))


def preparer(T, cfg):
    """Chargement v1 + features + _cat (+1 pour CatBoost) + one-hot éventuel : copie de 03_train."""
    train, test = U.charger_donnees(cfg["version"])
    feats = U.liste_features(train, cfg.get("exclure", []))
    cats = T.colonnes_cat(cfg["version"], feats)
    X, y, Xte = train[feats].astype(np.float32), train[U.TARGET].to_numpy(), test[feats].astype(np.float32)
    if cfg.get("cat_native") or cfg["modele"] == "catboost":
        for D in (X, Xte):
            D[cats] = D[cats] + 1
    if cfg.get("onehot"):
        X, Xte, cats = T.one_hot(X, Xte, cats, garder_origine=(cfg["onehot"] == "ajoute"))
    assert not cfg.get("te_cols"), "encodage cible non prévu dans le refit"
    return X, y, Xte, cats, test[U.ID].to_numpy()


def refit_arbres(exp: str):
    T = importlib.import_module("03_train")
    T.ajouter_configs_dynamiques("E02", "v1", [])        # définit E07/E08 comme dans la commande 2_306
    exp_cv = EQUIVALENT[exp]
    cfg = T.CONFIGS[exp_cv]
    X, y, Xte, cats, ids_te = preparer(T, cfg)
    if exp == "E80":
        n = int(round(cfg["n_max"] * FACTEUR))          # 1400 x 1,2 = 1680
    elif exp == "E85":
        n = int(round(1050 * FACTEUR))                   # 1050 (config E35, optimum CV) x 1,2 = 1260
    else:
        n = n_arbres_cv(exp_cv)
    print(f"[{exp}] refit {cfg['modele']} config {exp_cv} sur 100 % : X {X.shape}, {n} arbres", flush=True)
    t = time.time()
    if cfg["modele"] == "lightgbm":
        import lightgbm as lgb
        model = lgb.train(cfg["params"], lgb.Dataset(X, y), num_boost_round=n)
        p = model.predict(Xte)
    elif cfg["modele"] == "xgboost":
        import xgboost as xgb
        model = xgb.train(cfg["params"], xgb.QuantileDMatrix(X, y, max_bin=cfg["params"]["max_bin"]),
                          num_boost_round=n)
        p = model.predict(xgb.DMatrix(Xte))
    elif cfg["modele"] == "catboost":
        from catboost import CatBoostClassifier, Pool
        def pool(D, yy=None):
            D = D.copy()
            for c in cats:
                D[c] = D[c].astype(np.int64)
            return Pool(D, yy, cat_features=cats)
        model = CatBoostClassifier(**cfg["params"], iterations=n)   # pas d'early stopping
        model.fit(pool(X, y))
        p = model.predict_proba(pool(Xte))[:, 1]
    else:
        raise ValueError(cfg["modele"])
    print(f"[{exp}] entraînement + prédiction {time.time() - t:.0f}s", flush=True)
    return ids_te, p, n, time.time() - t


# ---------------------------------------------------------------------------
# 2. Réseau sur activations du DAE gelé (13_nn_dae.py), sans folds
# ---------------------------------------------------------------------------
def refit_reseau(graines=(0, 1, 2), n_ep=6, lr=1e-4, wd=1e-5, batch=256):
    import torch
    import torch.nn as nn
    N = importlib.import_module("13_nn_dae")
    X, y, fold, ids, ids_te, n_train, dae = N.charger(False, "dae.pt")
    i_tr = torch.arange(0, n_train, device=N.DEV)            # 100 % du train
    i_te = torch.arange(n_train, X.shape[0], device=N.DEV)
    yg = torch.from_numpy(y).to(N.DEV)
    n_in = N.representation(dae, X[:2]).shape[1]
    t0, preds = time.time(), []
    for g in graines:
        torch.manual_seed(U.SEED + 500 + 1000 * g)             # graine distincte des réseaux de fold
        model = N.Reseau(n_in).to(N.DEV)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        somme = 0.0
        for ep in range(1, n_ep + 1):                          # 6 époques fixes, comme E32b
            te = time.time(); model.train()
            perm = i_tr[torch.randperm(len(i_tr), device=N.DEV)]
            for i in range(0, len(perm), batch):
                b = perm[i:i + batch]
                perte = nn.functional.binary_cross_entropy_with_logits(model(N.representation(dae, X[b])), yg[b])
                opt.zero_grad(set_to_none=True); perte.backward(); opt.step()
            somme = somme + N.predire(model, dae, X, i_te)     # moyenne des prédictions des 6 époques
            print(f"  graine {g} ep {ep} | {time.time() - te:.1f}s", flush=True)
        p_g = somme / n_ep
        pd.DataFrame({U.ID: ids_te, "prediction": p_g}).to_csv(U.OOF_DIR / f"E83_g{g}_test.csv", index=False)
        preds.append(p_g)
    return ids_te, np.mean(preds, axis=0), n_ep, time.time() - t0


# ---------------------------------------------------------------------------
# 3. Sauvegarde + contrôle Spearman vs prédiction test par folds
# ---------------------------------------------------------------------------
def controler_et_sauver(exp, ids_te, p, n, duree):
    out = pd.DataFrame({U.ID: ids_te, "prediction": p}).sort_values(U.ID, ignore_index=True)
    assert out["prediction"].notna().all() and out["prediction"].between(0, 1).all()
    ref = pd.read_csv(U.OOF_DIR / f"{EQUIVALENT[exp]}_test.csv").sort_values(U.ID, ignore_index=True)
    assert (ref[U.ID].to_numpy() == out[U.ID].to_numpy()).all()
    rho = spearmanr(out["prediction"], ref["prediction"]).correlation
    out.to_csv(U.OOF_DIR / f"{exp}_test.csv", index=False)
    info = dict(exp=exp, equivalent=EQUIVALENT[exp], n=n, duree_s=round(duree, 1), spearman=round(float(rho), 5),
                moyenne=round(float(out["prediction"].mean()), 5),
                moyenne_equivalent=round(float(ref["prediction"].mean()), 5))
    (U.OOF_DIR / f"{exp}_refit_info.json").write_text(json.dumps(info), encoding="utf-8")
    print(f"[{exp}] RESULTAT {info}" + ("" if rho > 0.97 else "  !!! SPEARMAN ANORMALEMENT BAS"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EQUIVALENT))
    a = ap.parse_args()
    res = refit_reseau() if a.exp == "E83" else refit_arbres(a.exp)
    controler_et_sauver(a.exp, *res)
