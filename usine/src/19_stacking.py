"""19_stacking.py — stacking par méta-modèle (idée du professeur : pondérer les modèles selon le profil).

Entrées (aucun réentraînement des modèles de base) :
  - OOF et test des 4 membres de 2_306 / E33f : E30 (LightGBM), E07 (XGBoost), E08 (CatBoost), E32c (réseau DAE)
  - 6 variables de profil tirées de v1 : nb_manquants, ps_car_13, ps_ind_03, ps_reg_03, ps_ind_05_cat (==0), ps_ind_17_bin
Prédictions des membres -> rangs percentiles, calculés SÉPARÉMENT sur l'OOF et sur le test
(les prédictions test sont des moyennes de 5 modèles : échelle différente de l'OOF).

Évaluation emboîtée sur les folds figés : pour chaque fold k, le méta-modèle est ajusté sur l'OOF
des 4 autres folds (y compris centrage/réduction et imputation), puis prédit le fold k -> méta-OOF.
Comparaison appariée vs E33f (moyenne des rangs à poids égaux) avec 03_train.comparer.

Méta-modèles, fixés AVANT de voir les résultats :
  E90  régression logistique sur les 4 rangs (poids appris, constants)
  E91  régression logistique rangs + profil + interactions rang x profil (poids dépendant du profil)
  E92  LightGBM peu profond, très régularisé, rangs + profil, 300 arbres fixes
Le meilleur (Gini méta-OOF) est réentraîné sur tout l'OOF et appliqué au test ; remise à l'échelle
des probabilités par quantile mapping (valeurs triées de la moyenne des probas des membres).

Usage : python -u usine/src/19_stacking.py --sortie usine/submissions/2_310.csv
"""
import argparse, importlib.util, sys, time, pathlib
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U
spec = importlib.util.spec_from_file_location("t03", pathlib.Path(__file__).parent / "03_train.py")
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)  # pour comparer()

ap = argparse.ArgumentParser()
ap.add_argument("--membres", nargs="+", default=["E30", "E07", "E08", "E32c"])
ap.add_argument("--ref", default="E33f")
ap.add_argument("--probas", nargs="+", default=["E30", "E07", "E08", "E32b"],
                help="membres dont la moyenne des probas donne l'échelle (E32c_test est en rangs -> E32b, comme 07)")
ap.add_argument("--sans-registre", action="store_true", help="ne réécrit pas E90-E92 (relance)")
ap.add_argument("--sortie", default="usine/submissions/2_310.csv")
ap.add_argument("--reference-sub", default="usine/submissions/2_306.csv")
a = ap.parse_args()
t0 = time.time()
M = a.membres
PROFIL = ["nb_manquants", "ps_car_13", "ps_ind_03", "ps_reg_03", "ind05_zero", "ps_ind_17_bin"]

# ---------------------------------------------------------------- 1. Données
train, test = U.charger_donnees("v1")
fold = U.charger_folds(train)
y = train[U.TARGET].to_numpy()

def profil(df):
    feats = U.liste_features(df)
    X = df[feats]
    p = pd.DataFrame({"nb_manquants": ((X == -1) | X.isna()).sum(axis=1).astype(float)})
    for c in ["ps_car_13", "ps_ind_03", "ps_reg_03", "ps_ind_17_bin"]:
        p[c] = df[c].replace(-1, np.nan).astype(float)
    p["ind05_zero"] = (df["ps_ind_05_cat"] == 0).astype(float)
    return p[PROFIL]

P_tr, P_te = profil(train), profil(test)

def rangs(prefix, suffixe, col, ids):
    out = {}
    for m in M:
        d = pd.read_csv(U.OOF_DIR / f"{m}_{suffixe}.csv")
        d = d.set_index(U.ID).loc[ids]
        out[m] = d[col].to_numpy()
    return pd.DataFrame({f"r_{m}": rankdata(v) / len(v) for m, v in out.items()}), out

R_tr, _ = rangs("oof", "oof", "oof", train[U.ID].to_numpy())
R_te, probas_te = rangs("test", "test", "prediction", test[U.ID].to_numpy())
print(f"[données] train {R_tr.shape}, test {R_te.shape}, profil {PROFIL}", flush=True)

# ---------------------------------------------------------------- 2. Construction des matrices
def matrices(R_a, P_a, R_b, P_b, avec_profil, interactions):
    """Prépare X (apprentissage) et X (application) ; statistiques calculées sur 'a' uniquement."""
    Ra, Rb = R_a.to_numpy() - 0.5, R_b.to_numpy() - 0.5           # rangs centrés
    if not avec_profil:
        return Ra, Rb, list(R_a.columns)
    med = P_a.median()
    Pa, Pb = P_a.fillna(med), P_b.fillna(med)
    mu, sd = Pa.mean(), Pa.std().replace(0, 1)
    Pa, Pb = ((Pa - mu) / sd).to_numpy(), ((Pb - mu) / sd).to_numpy()
    Xa, Xb, noms = [Ra, Pa], [Rb, Pb], list(R_a.columns) + PROFIL
    if interactions:
        for i, m in enumerate(R_a.columns):
            for j, p in enumerate(PROFIL):
                Xa.append(Ra[:, [i]] * Pa[:, [j]]); Xb.append(Rb[:, [i]] * Pb[:, [j]])
                noms.append(f"{m}x{p}")
    return np.hstack(Xa), np.hstack(Xb), noms

def ajuster_predire(exp, Xa, ya, Xb):
    if exp in ("E90", "E91"):
        mdl = LogisticRegression(C=1.0, max_iter=2000)
        mdl.fit(Xa, ya); return mdl.predict_proba(Xb)[:, 1], mdl
    params = dict(objective="binary", num_leaves=6, min_data_in_leaf=2000, learning_rate=0.02,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, num_threads=12, seed=42)
    mdl = lgb.train(params, lgb.Dataset(Xa, ya), num_boost_round=300)
    return mdl.predict(Xb), mdl

CONFIGS = {"E90": (False, False), "E91": (True, True), "E92": (True, False)}
DESC = {"E90": "méta logistique sur rangs des 4 membres (poids appris)",
        "E91": "méta logistique rangs + 6 profils + interactions rang x profil",
        "E92": "méta LightGBM (6 feuilles, min 2000, lr 0,02, 300 arbres) rangs + 6 profils"}

# ---------------------------------------------------------------- 3. Évaluation emboîtée
res = {}
for exp, (avec_p, inter) in CONFIGS.items():
    oof, ginis = np.zeros(len(y)), []
    for k in range(U.N_FOLDS):
        tr, va = fold != k, fold == k
        Xa, Xb, _ = matrices(R_tr[tr], P_tr[tr], R_tr[va], P_tr[va], avec_p, inter)
        oof[va], _ = ajuster_predire(exp, Xa, y[tr], Xb)
        ginis.append(U.gini(y[va], oof[va]))
    g = np.array(ginis)
    cmp = T.comparer(exp, a.ref, g, enregistrer=not a.sans_registre)
    res[exp] = dict(oof=oof, g=g, cmp=cmp)
    pd.DataFrame({U.ID: train[U.ID], U.TARGET: y, "oof": oof, "fold": fold}).to_csv(U.OOF_DIR / f"{exp}_oof.csv", index=False, lineterminator="\n")
    pd.DataFrame({"fold": range(5), "gini": g}).to_csv(U.OOF_DIR / f"{exp}_folds_detail.csv", index=False, lineterminator="\n")
    print(f"[{exp}] Gini {g.mean():.5f} ± {g.std():.5f} | folds {np.round(g,5)} | vs {a.ref}: Δ {cmp['d_moy']:+.5f}, "
          f"{cmp['folds_gagnes']} gagnés, seuil 2se {cmp['seuil_2se']} -> {cmp['verdict']} ({time.time()-t0:.0f}s)", flush=True)
    if not a.sans_registre: U.ajouter_registre(dict(id=exp, donnees_version="v1 (OOF membres + profil)", modele="stacking",
        changement=DESC[exp], score_cv_moyen=g.mean(), score_cv_std=g.std(), delta_vs_ref=cmp["d_moy"],
        verdict=cmp["verdict"], commentaire=f"emboîté sur folds figés; vs {a.ref}: folds gagnés {cmp['folds_gagnes']}, "
        f"Δ/fold {[cmp[f'd_fold{k}'] for k in range(5)]}; membres {'+'.join(M)}"))

# ---------------------------------------------------------------- 4. Interprétation E91 (modèle sur tout l'OOF)
Xa, _, noms = matrices(R_tr, P_tr, R_te, P_te, True, True)
_, lr91 = ajuster_predire("E91", Xa, y, Xa[:5])
coef = pd.Series(lr91.coef_[0], index=noms)
print("\n[E91] coefficients (profils standardisés, rangs centrés) :")
print(coef.round(4).to_string())
print("\n[E91] poids effectif de chaque membre selon nb_manquants (-1 sd / 0 / +1 sd) :")
for m in M:
    b, i = coef[f"r_{m}"], coef[f"r_{m}xnb_manquants"]
    print(f"  {m}: {b-i:.3f} / {b:.3f} / {b+i:.3f}")
coef.to_csv(U.EXP_DIR / "E91_coefficients.csv", header=["coef"], lineterminator="\n")

# ---------------------------------------------------------------- 5. Meilleur -> test -> soumission
best = max(res, key=lambda e: res[e]["g"].mean())
avec_p, inter = CONFIGS[best]
Xa, Xb, _ = matrices(R_tr, P_tr, R_te, P_te, avec_p, inter)
pred_rang, _ = ajuster_predire(best, Xa, y, Xb)
pd.DataFrame({U.ID: test[U.ID], "prediction": pred_rang}).to_csv(U.OOF_DIR / f"{best}_test.csv", index=False, lineterminator="\n")
ids_te = test[U.ID].to_numpy()
moy_proba = np.mean([pd.read_csv(U.OOF_DIR / f"{m}_test.csv").set_index(U.ID).loc[ids_te, "prediction"].to_numpy() for m in a.probas], axis=0)
ordre = np.argsort(pred_rang, kind="mergesort")
pred = np.empty_like(moy_proba); pred[ordre] = np.sort(moy_proba)
ids_csv = pd.read_csv(U.TEST_CSV, usecols=[U.ID])[U.ID].to_numpy()
assert (ids_csv == test[U.ID].to_numpy()).all(), "ordre test.csv != test v1 trié"
U.ecrire_soumission(test[U.ID], pred, a.sortie)
U.verifier_soumission(a.sortie, float(y.mean()))
ref = pd.read_csv(a.reference_sub, header=None, names=[U.ID, "p"])
rho = spearmanr(ref["p"], pred).correlation
print(f"\n[soumission] méta retenu = {best} ; Spearman vs {a.reference_sub} = {rho:.5f}"
      + ("  ALERTE < 0,9" if rho < 0.9 else "") + f" ; moyenne {pred.mean():.5f} ; {time.time()-t0:.0f}s")
U.ajouter_registre(dict(id="E93", donnees_version="v1", modele="stacking", changement=f"{best} réentraîné sur tout l'OOF -> {a.sortie}",
    score_cv_moyen=res[best]["g"].mean(), score_cv_std=res[best]["g"].std(), delta_vs_ref="n/a", verdict="info",
    commentaire=f"Spearman vs 2_306 {rho:.5f}; moyenne {pred.mean():.5f}; format OK (148803 lignes, sans en-tête, ordre test.csv, LF, 0 NaN)"))
