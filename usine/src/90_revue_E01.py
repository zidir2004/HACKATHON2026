"""90_revue_E01.py — contrôle reviewer (n'écrit NI registre NI oof).
a) reproduction exacte de E01 (mêmes folds) ; b) nombre d'arbres fixe (600, sans early stopping)
c) E01 avec early stopping sur 2 autres découpages (graines 7 et 2026)."""
import sys, pathlib, importlib.util
import numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U
spec = importlib.util.spec_from_file_location("tr", pathlib.Path(__file__).parent / "03_train.py")
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
P = tr.LGB_BASE
train, test = U.charger_donnees("v0")
feats = U.liste_features(train); assert "id" not in feats and "target" not in feats
X = train[feats].astype(np.float32); y = train[U.TARGET].to_numpy()
def cv(fold, fixe=None):
    g, oof = [], np.zeros(len(y))
    for k in range(5):
        a, b = fold != k, fold == k
        if fixe:
            m = lgb.train(P, lgb.Dataset(X[a], y[a]), num_boost_round=fixe); p = m.predict(X[b])
        else:
            m = lgb.train(P, lgb.Dataset(X[a], y[a]), 3000, valid_sets=[lgb.Dataset(X[b], y[b])],
                          callbacks=[lgb.early_stopping(200, verbose=False)]); p = m.predict(X[b], num_iteration=m.best_iteration)
        oof[b] = p; g.append(U.gini(y[b], p))
    return np.mean(g), np.std(g), U.gini(y, oof), [round(x, 5) for x in g]
f0 = U.charger_folds(train)
print("a) repro E01 (ES)      :", cv(f0), flush=True)
print("b) 600 arbres fixes    :", cv(f0, 600), flush=True)
for s in (7, 2026):
    f = np.zeros(len(y), int)
    for k, (_, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)): f[v] = k
    print(f"c) graine {s} ES       :", cv(f), flush=True)
    print(f"c) graine {s} 600 fixes:", cv(f, 600), flush=True)
