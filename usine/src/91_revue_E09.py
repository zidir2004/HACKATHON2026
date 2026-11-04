"""91_revue_E09.py - controle reviewer (n'ecrit NI registre NI oof).
Rejoue E02 (LGB), E07 (XGB GPU), E08 (CatBoost GPU) et le melange rangs poids egaux
avec un NOMBRE D'ARBRES FIXE (pas d'early stopping sur le fold rapporte) sur le decoupage --seed.
Usage : python -u usine/src/91_revue_E09.py --seed 7"""
import sys, pathlib, importlib.util, argparse, time
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U
spec = importlib.util.spec_from_file_location("tr", pathlib.Path(__file__).parent / "03_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", nargs=3, type=int, default=[600, 400, 850]); a = ap.parse_args()
train, test = U.charger_donnees("v1")
feats = U.liste_features(train); cats = T.colonnes_cat("v1", feats)
X = train[feats].astype(np.float32); y = train[U.TARGET].to_numpy()
Xc = X.copy(); Xc[cats] = Xc[cats] + 1
if a.seed == 42:
    fold = U.charger_folds(train)
else:
    fold = np.zeros(len(y), int)
    for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=a.seed).split(X, y)): fold[v] = k
print(f"seed {a.seed}, {len(feats)} feats, arbres fixes LGB/XGB/CB = {a.n}", flush=True)
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
def pool(D, yy=None):
    D = D.copy()
    for c in cats: D[c] = D[c].astype(np.int64)
    return Pool(D, yy, cat_features=cats)
oof = {m: np.zeros(len(y)) for m in ("lgb", "xgb", "cb")}
for k in range(5):
    tr, va = fold != k, fold == k; t = time.time()
    m = lgb.train(T.LGB_BASE, lgb.Dataset(X[tr], y[tr]), num_boost_round=a.n[0]); oof["lgb"][va] = m.predict(X[va])
    p = {k2: v for k2, v in T.XGB_BASE.items()}
    d = xgb.QuantileDMatrix(X[tr], y[tr], max_bin=p["max_bin"])
    m = xgb.train(p, d, num_boost_round=a.n[1]); oof["xgb"][va] = m.predict(xgb.DMatrix(X[va]))
    m = CatBoostClassifier(**T.CB_BASE, iterations=a.n[2]); m.fit(pool(Xc[tr], y[tr]))
    oof["cb"][va] = m.predict_proba(pool(Xc[va]))[:, 1]
    print(f"fold {k} : " + " ".join(f"{n}={U.gini(y[va], o[va]):.5f}" for n, o in oof.items()) + f" ({time.time()-t:.0f}s)", flush=True)
def gf(p): return np.array([U.gini(y[fold == k], p[fold == k]) for k in range(5)])
R = np.column_stack([rankdata(oof[m]) for m in ("lgb", "xgb", "cb")]); bl = R.mean(1)
res = {n: gf(o) for n, o in oof.items()}; res["melange"] = gf(bl)
for n, g in res.items():
    print(f"{n:8s} CV {g.mean():.5f} +- {g.std():.5f} | {np.round(g, 5).tolist()}")
d = res["melange"] - res["lgb"]
print(f"melange - lgb : dmoy {d.mean():+.5f} gagnes {(d>0).sum()}/5 {np.round(d,5).tolist()}")
d = res["melange"] - np.max([res[m] for m in ("lgb", "xgb", "cb")], axis=0)
print(f"melange - meilleur membre par fold : dmoy {d.mean():+.5f} gagnes {(d>0).sum()}/5")
