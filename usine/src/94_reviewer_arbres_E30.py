"""Reviewer 94 : courbe Gini du LightGBM E30 sur les 5 folds jusqu'a 1680 arbres (verifie le x1,2 de E80)."""
import importlib, pathlib, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U
T = importlib.import_module("03_train")
R = importlib.import_module("17_refit_complet")
cfg = T.CONFIGS["E30"]
X, y, Xte, cats, _ = R.preparer(T, cfg)
train, _ = U.charger_donnees(cfg["version"])
folds = pd.read_csv(U.FOLDS_CSV).set_index(U.ID).loc[train[U.ID], "fold"].to_numpy()
pts = [800, 1000, 1200, 1400, 1680]
res = []
for k in range(5):
    tr, va = folds != k, folds == k
    m = lgb.train(cfg["params"], lgb.Dataset(X[tr], y[tr]), num_boost_round=1680)
    g = [2 * roc_auc_score(y[va], m.predict(X[va], num_iteration=n)) - 1 for n in pts]
    res.append(g); print(k, [round(v, 5) for v in g], flush=True)
r = np.array(res)
print("moyenne", dict(zip(pts, np.round(r.mean(0), 5))))
print("delta 1680-1400 par fold", np.round(r[:, 4] - r[:, 3], 5))
