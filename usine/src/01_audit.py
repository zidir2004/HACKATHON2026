"""01_audit.py - Audit des donnees (LECTURE SEULE : aucun fichier d'origine n'est modifie).

Lance depuis le dossier Hackathon :  python -u usine/src/01_audit.py
Sections :
  1. Chargement, typage, sauvegarde parquet (non corriges)
  2. Profil par variable (manquants -1, min/max, taux de sinistre manquant vs present)
  3. Signal univarie (Gini brut, Gini de l'encodage cible en CV 5 folds, bruit de reference par permutation)
  4. Derive train/test (KS, PSI, adversarial validation LightGBM)
  5. Doublons, correlations, constantes, codes, colonnes deduites, structure des variables continues
  6. Figures PNG (prefixe 01_)
  7. Sauvegardes (CSV + resume JSON) pour rediger le rapport
Le rapport .md est redige a part, a partir du log et des CSV.
"""
import json, time, traceback, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(".")
INTERIM, FIG, RAP = ROOT / "usine/data/interim", ROOT / "usine/figures", ROOT / "usine/rapports"
for d in (INTERIM, FIG, RAP):
    d.mkdir(parents=True, exist_ok=True)
SEED, NJOBS = 42, 16
T0 = time.time()
RES = {}          # resume chiffre (sauve en JSON)
FAILED = []       # sections en erreur


def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def show(title, df, n=60):
    print(f"\n--- {title} ---")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(df.head(n).to_string())


def var_group(c): return c.split("_")[1]                                   # ind / reg / car / calc
def var_kind(c): return "bin" if c.endswith("_bin") else ("cat" if c.endswith("_cat") else "num")
def gini(y_true, score): return 2 * roc_auc_score(y_true, score) - 1


# =====================================================================
# 1. CHARGEMENT ET TYPAGE
# =====================================================================
log("1. chargement des fichiers")
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)
sub = pd.read_csv("submission_example.csv", header=None, names=["id", "prediction"])
FEATS = [c for c in train.columns if c not in ("id", "target")]
RES["train_shape"], RES["test_shape"], RES["sub_shape"] = list(train.shape), list(test.shape), list(sub.shape)
RES["memes_colonnes_train_test"] = bool(set(FEATS) == set(c for c in test.columns if c != "id"))
RES["ordre_colonnes_identique"] = bool(list(test.columns) == ["id"] + FEATS)
test = test[["id"] + FEATS]                                                # reordonne en memoire seulement
RES["nan_train"], RES["nan_test"] = int(train.isna().sum().sum()), int(test.isna().sum().sum())
RES["id_doublons_train"], RES["id_doublons_test"] = int(train["id"].duplicated().sum()), int(test["id"].duplicated().sum())
RES["id_communs_train_test"] = int(np.intersect1d(train["id"], test["id"]).size)
RES["test_trie_croissant"] = bool(test["id"].is_monotonic_increasing)
RES["train_trie_croissant"] = bool(train["id"].is_monotonic_increasing)
RES["sub_ids_egaux_test"] = bool(len(sub) == len(test) and np.array_equal(sub["id"].to_numpy(), test["id"].to_numpy()))
RES["sub_prediction_valeurs"] = sorted(sub["prediction"].unique().tolist())[:5]
RES["cible_valeurs"] = sorted(train["target"].unique().tolist())
RES["id_min_max_train"] = [int(train.id.min()), int(train.id.max())]
RES["id_min_max_test"] = [int(test.id.min()), int(test.id.max())]
y = train["target"].to_numpy().astype(np.int8)
N, N1 = len(y), int(y.sum())
BASE = y.mean()
RES["taux_cible"], RES["n_positifs"] = float(BASE), N1

# copies float64 des donnees d'origine pour toutes les analyses
tr64 = train[FEATS].astype("float64")
te64 = test[FEATS].astype("float64")

# types optimises : int8/int16/int32 si valeurs entieres, sinon float32 (meme type train/test)
both = pd.concat([train[FEATS], test[FEATS]], ignore_index=True)
DTYPES = {}
for c in FEATS:
    v = both[c].to_numpy().astype("float64")
    if np.array_equal(v, np.round(v)):
        lo, hi = v.min(), v.max()
        DTYPES[c] = "int8" if (lo >= -128 and hi <= 127) else ("int16" if (lo >= -32768 and hi <= 32767) else "int32")
    else:
        DTYPES[c] = "float32"
del both
train_t = pd.concat([train[["id"]].astype("int32"), train[["target"]].astype("int8"),
                     train[FEATS].astype(DTYPES)], axis=1)
test_t = pd.concat([test[["id"]].astype("int32"), test[FEATS].astype(DTYPES)], axis=1)
train_t.to_parquet(INTERIM / "train_raw.parquet", index=False)
test_t.to_parquet(INTERIM / "test_raw.parquet", index=False)
# controle : le typage ne change aucune valeur (ecart max float32 vs float64)
fl = [c for c in FEATS if DTYPES[c] == "float32"]
RES["ecart_max_float32"] = float(max(np.abs(train_t[c].astype("float64") - tr64[c]).max() for c in fl))
RES["ecart_max_entiers"] = float(max(np.abs(train_t[c].astype("float64") - tr64[c]).max() for c in FEATS if DTYPES[c] != "float32"))
RES["dtypes"] = pd.Series(DTYPES).value_counts().to_dict()
RES["mem_train_typee_Mo"] = round(train_t.memory_usage(deep=True).sum() / 2**20, 1)
RES["mem_train_brute_Mo"] = round(train.memory_usage(deep=True).sum() / 2**20, 1)
log(f"parquet ecrits. train {train.shape}, test {test.shape}; dtypes {RES['dtypes']}")
log(f"controles: {json.dumps({k: RES[k] for k in ['memes_colonnes_train_test','ordre_colonnes_identique','nan_train','nan_test','id_doublons_train','id_communs_train_test','test_trie_croissant','sub_ids_egaux_test']})}")

# =====================================================================
# 2. PROFIL PAR VARIABLE
# =====================================================================
log("2. profil des variables")
rows = []
for c in FEATS:
    a, b = tr64[c].to_numpy(), te64[c].to_numpy()
    miss = a == -1
    nm = int(miss.sum())
    r = dict(var=c, groupe=var_group(c), type=var_kind(c), dtype=DTYPES[c],
             nunique_train=int(pd.Series(a).nunique()), nunique_test=int(pd.Series(b).nunique()),
             n_manquants_train=nm, pct_manq_train=100 * nm / N, pct_manq_test=100 * float((b == -1).mean()),
             min=float(a.min()), max=float(a.max()),
             min_hors_manq=float(a[~miss].min()), max_hors_manq=float(a[~miss].max()),
             moy_hors_manq=float(a[~miss].mean()), ecart_type_hors_manq=float(a[~miss].std()),
             part_modalite_majoritaire=float(pd.Series(a).value_counts(normalize=True).iloc[0]))
    if nm > 0:
        p1, p0 = y[miss].mean(), y[~miss].mean()
        pp = y.mean(); se = np.sqrt(pp * (1 - pp) * (1 / nm + 1 / (N - nm)))
        r.update(taux_cible_manquant=float(p1), taux_cible_present=float(p0), lift_manquant=float(p1 / p0),
                 z_manquant=float((p1 - p0) / se))
    rows.append(r)
prof = pd.DataFrame(rows).set_index("var")
RES["variables_constantes"] = prof.index[prof.nunique_train <= 1].tolist()
RES["variables_avec_manquants"] = prof.index[prof.n_manquants_train > 0].tolist()
# nombre de -1 par ligne
nmiss_row = (tr64 == -1).sum(axis=1).to_numpy()
RES["gini_nb_manquants_par_ligne"] = float(gini(y, nmiss_row))
RES["taux_cible_par_nb_manquants"] = pd.Series(y).groupby(nmiss_row).agg(["size", "mean"]).round(4).to_dict("index")
RES["manquants_par_groupe"] = {g: dict(n_vars=int(len(d)), n_vars_avec_manq=int((d.n_manquants_train > 0).sum()),
                                       pct_cellules_manq=round(float(100 * d.n_manquants_train.sum() / (len(d) * N)), 3))
                               for g, d in prof.groupby("groupe")}
show("Variables avec manquants", prof[prof.n_manquants_train > 0][
    ["groupe", "pct_manq_train", "pct_manq_test", "taux_cible_manquant", "taux_cible_present", "lift_manquant", "z_manquant"]]
    .sort_values("pct_manq_train", ascending=False))
print("Gini du nombre de -1 par ligne:", RES["gini_nb_manquants_par_ligne"])
print("Taux cible par nb de -1 par ligne:", RES["taux_cible_par_nb_manquants"])

# blocs de manquants : correlation entre indicateurs de manquants
mv = RES["variables_avec_manquants"]
M = (tr64[mv] == -1).astype(float)
Mc = M.corr()
blocs = [(a, b, float(Mc.loc[a, b])) for i, a in enumerate(mv) for b in mv[i + 1:] if abs(Mc.loc[a, b]) > 0.3]
blocs = sorted(blocs, key=lambda t: -abs(t[2]))
RES["blocs_manquants_corr>0.3"] = [(a, b, round(r, 3)) for a, b, r in blocs]
print("Paires de variables manquantes ensemble (|r|>0.3):", RES["blocs_manquants_corr>0.3"])

# =====================================================================
# 3. SIGNAL UNIVARIE
# =====================================================================
log("3. signal univarie")
ranks = np.column_stack([rankdata(tr64[c].to_numpy()) for c in FEATS])          # rangs moyens (ex aequo geres)
pos = y == 1
n1, n0 = N1, N - N1
g_from_rank = lambda s: 2 * ((s - n1 * (n1 + 1) / 2) / (n1 * n0)) - 1
gini_raw = g_from_rank(ranks[pos].sum(axis=0))                                  # Gini brut, -1 laisse tel quel
# bruit de reference : on melange la cible 300 fois -> distribution du Gini "pour rien"
rng = np.random.default_rng(SEED)
NP = 300
null = np.empty((NP, len(FEATS)))
for i in range(NP):
    null[i] = g_from_rank(ranks[rng.choice(N, n1, replace=False)].sum(axis=0))
null_sd = null.std(axis=0)
thr_var = float(np.quantile(np.abs(null), 0.95))                                # seuil 5 % pour 1 variable
thr_fam = float(np.quantile(np.abs(null).max(axis=1), 0.95))                    # seuil 5 % sur 57 variables a la fois
RES["bruit_gini_ecart_type_moyen"], RES["seuil_gini_5pct_1var"], RES["seuil_gini_5pct_57vars"] = float(null_sd.mean()), thr_var, thr_fam
del ranks

sig = pd.DataFrame(index=FEATS)
sig["groupe"] = [var_group(c) for c in FEATS]
sig["type"] = [var_kind(c) for c in FEATS]
sig["gini_brut"] = gini_raw
sig["abs_gini_brut"] = np.abs(gini_raw)
sig["z_vs_bruit"] = gini_raw / null_sd
gh, gm = [], []
for c in FEATS:
    a = tr64[c].to_numpy(); miss = a == -1
    gm.append(float(gini(y, miss.astype(int))) if 0 < miss.sum() < N else np.nan)
    gh.append(float(gini(y[~miss], a[~miss])) if 0 < miss.sum() < N - 100 and np.unique(a[~miss]).size > 1 else np.nan)
sig["gini_hors_manquants"] = gh
sig["gini_indicateur_manquant"] = gm

folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(N), y))


def to_codes(s):
    """Codes entiers 0..K-1 : modalites telles quelles si <=120, sinon 50 quantiles (+ une classe pour -1)."""
    v = s.to_numpy()
    if np.unique(v).size <= 120:
        return pd.factorize(v)[0]
    miss = v == -1
    q = pd.qcut(pd.Series(np.where(miss, np.nan, v)), 50, duplicates="drop")
    codes = q.cat.codes.to_numpy().copy()
    codes[miss] = codes.max() + 1
    return codes


def te_oof(codes, yy, m=50):
    """Taux de cible lisse, calcule sur les 4 autres folds (jamais sur le fold qu'on encode)."""
    K = codes.max() + 1
    enc = np.zeros(len(yy))
    for tr, va in folds:
        cnt = np.bincount(codes[tr], minlength=K)
        ps = np.bincount(codes[tr], weights=yy[tr], minlength=K)
        enc[va] = ((ps + m * yy[tr].mean()) / (cnt + m))[codes[va]]
    return enc


CODES = {c: to_codes(tr64[c]) for c in FEATS}
te_p, te_m, te_s = [], [], []
for c in FEATS:
    enc = te_oof(CODES[c], y.astype(float))
    te_p.append(gini(y, enc))
    fg = [gini(y[va], enc[va]) for _, va in folds]
    te_m.append(np.mean(fg)); te_s.append(np.std(fg))
# NB: le Gini "poole" sur les 5 folds est biaise (la moyenne du train change d'un fold a l'autre) ; on retient la moyenne des Gini par fold
sig["gini_te_pooled"], sig["gini_te"], sig["gini_te_fold_std"] = te_p, te_m, te_s
# bruit de l'encodage cible lui-meme (cible melangee, 5 tirages)
tn = np.zeros((5, len(FEATS)))
for k in range(5):
    yp = np.random.default_rng(100 + k).permutation(y).astype(float)
    for j, c in enumerate(FEATS):
        enc = te_oof(CODES[c], yp)
        tn[k, j] = np.mean([gini(yp[va], enc[va]) for _, va in folds])
sig["bruit_te_abs_moy"] = np.abs(tn).mean(axis=0)
sig["signal"] = np.maximum(sig.abs_gini_brut, sig.gini_te.clip(lower=0))      # meilleur des deux mesures
sig["rang"] = sig.signal.rank(ascending=False).astype(int)
RES["bruit_te_abs_max"] = float(np.abs(tn).max())
sig = sig.sort_values("signal", ascending=False)
show("Top 35 signal univarie", sig[["groupe", "type", "gini_brut", "gini_te", "gini_te_fold_std", "bruit_te_abs_moy",
                                     "gini_hors_manquants", "gini_indicateur_manquant", "z_vs_bruit"]], 35)
print(f"\nBruit: sd moyenne du Gini sous H0 = {null_sd.mean():.4f} ; seuil 5% (1 var) = {thr_var:.4f} ; seuil 5% (57 vars) = {thr_fam:.4f}; bruit max TE (5 permutations) = {RES['bruit_te_abs_max']:.4f}")
grp = sig.groupby("groupe").agg(n=("signal", "size"), signal_moy=("signal", "mean"), signal_max=("signal", "max"),
                                gini_brut_abs_moy=("abs_gini_brut", "mean"), te_moy=("gini_te", "mean"),
                                nb_au_dessus_seuil57=("abs_gini_brut", lambda s: int((s > thr_fam).sum())),
                                nb_te_au_dessus_bruit_max=("gini_te", lambda s: int((s > RES["bruit_te_abs_max"]).sum())))
show("Signal par groupe", grp)
RES["signal_par_groupe"] = grp.round(4).to_dict("index")
calc = sig[sig.groupe == "calc"]
RES["calc_max_abs_gini_brut"], RES["calc_max_gini_te"] = float(calc.abs_gini_brut.max()), float(calc.gini_te.max())
RES["calc_variables_au_dessus_seuil57"] = calc.index[calc.abs_gini_brut > thr_fam].tolist()
RES["calc_variables_te_au_dessus_bruit"] = calc.index[calc.gini_te > RES["bruit_te_abs_max"]].tolist()
# fuites potentielles : AUC univariee > 0.9 (Gini > 0.8) + identifiant
RES["fuites_gini_brut>0.8"] = sig.index[sig.abs_gini_brut > 0.8].tolist()
RES["gini_id_vs_cible"] = float(gini(y, train["id"].to_numpy()))
dec = pd.qcut(train["id"], 10, labels=False)
RES["taux_cible_par_decile_id"] = pd.Series(y).groupby(dec.to_numpy()).mean().round(4).tolist()
print("Gini de l'id vs cible:", RES["gini_id_vs_cible"], "| taux cible par decile d'id:", RES["taux_cible_par_decile_id"])

# taux de cible par modalite (variables a <=120 valeurs)
mod_rows = []
for c in FEATS:
    a = tr64[c]
    if a.nunique() <= 120:
        g = pd.DataFrame({"v": a, "y": y}).groupby("v")["y"].agg(["size", "mean"])
        for v, r in g.iterrows():
            mod_rows.append((c, v, int(r["size"]), r["size"] / N, r["mean"], r["mean"] / BASE))
pd.DataFrame(mod_rows, columns=["var", "valeur", "n", "part", "taux_cible", "lift"]).to_csv(RAP / "audit_modalites.csv", index=False)

# Les ps_calc_* ont-ils un signal COMBINE ? LightGBM 5 folds sur 3 jeux de variables
log("3b. LightGBM: toutes / sans calc / calc seuls")
Xtr = train_t[FEATS].to_numpy(dtype=np.float32)
lgb_params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, n_jobs=NJOBS, verbose=-1, random_state=SEED)


def cv_lgb(cols):
    idx = [FEATS.index(c) for c in cols]
    oof, fg = np.zeros(N), []
    for tr, va in folds:
        m = lgb.LGBMClassifier(**lgb_params).fit(Xtr[tr][:, idx], y[tr])
        oof[va] = m.predict_proba(Xtr[va][:, idx])[:, 1]
        fg.append(gini(y[va], oof[va]))
    return float(np.mean(fg)), float(np.std(fg)), float(gini(y, oof))


nocalc = [c for c in FEATS if var_group(c) != "calc"]
calc_only = [c for c in FEATS if var_group(c) == "calc"]
lg = {}
for name, cols in [("toutes", FEATS), ("sans_calc", nocalc), ("calc_seuls", calc_only)]:
    lg[name] = cv_lgb(cols)
    log(f"   LGBM {name}: Gini fold moyen {lg[name][0]:.4f} +/- {lg[name][1]:.4f} (pooled {lg[name][2]:.4f})")
RES["lgbm_gini_cv"] = {k: dict(moy=v[0], std=v[1], pooled=v[2]) for k, v in lg.items()}

# =====================================================================
# 4. DERIVE TRAIN / TEST
# =====================================================================
try:
    log("4. derive train/test")

    def psi_series(a, b):
        eps = 1e-6
        if np.unique(a).size <= 120:
            la, lb = a, b
        else:
            edges = np.unique(np.quantile(a[a != -1], np.linspace(0, 1, 11)))[1:-1]
            la = np.where(a == -1, -1, np.searchsorted(edges, a))
            lb = np.where(b == -1, -1, np.searchsorted(edges, b))
        pa = pd.Series(la).value_counts(normalize=True)
        pb = pd.Series(lb).value_counts(normalize=True)
        idx = pa.index.union(pb.index)
        pa, pb = pa.reindex(idx).fillna(0) + eps, pb.reindex(idx).fillna(0) + eps
        return float(((pa - pb) * np.log(pa / pb)).sum())

    dr = []
    for c in FEATS:
        a, b = tr64[c].to_numpy(), te64[c].to_numpy()
        ks = ks_2samp(a, b)
        dr.append(dict(var=c, moy_train=a.mean(), moy_test=b.mean(), ks=ks.statistic, ks_p=ks.pvalue,
                       psi=psi_series(a, b), pct_manq_train=100 * (a == -1).mean(), pct_manq_test=100 * (b == -1).mean()))
    drift = pd.DataFrame(dr).set_index("var")
    show("Derive : plus grands PSI / KS", drift.sort_values("psi", ascending=False), 12)
    RES["derive_psi_max"], RES["derive_ks_max"] = float(drift.psi.max()), float(drift.ks.max())
    RES["derive_nb_psi>0.1"] = int((drift.psi > 0.1).sum())
    RES["derive_ecart_max_pct_manq"] = float((drift.pct_manq_train - drift.pct_manq_test).abs().max())

    Xa = np.vstack([Xtr, test_t[FEATS].to_numpy(dtype=np.float32)])
    ya = np.r_[np.zeros(N), np.ones(len(test_t))].astype(int)
    fa = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya))
    oof_a, imp = np.zeros(len(ya)), np.zeros(len(FEATS))
    pa_ = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, n_jobs=NJOBS, verbose=-1, random_state=SEED)
    for tr, va in fa:
        m = lgb.LGBMClassifier(**pa_).fit(Xa[tr], ya[tr])
        oof_a[va] = m.predict_proba(Xa[va])[:, 1]
        imp += m.booster_.feature_importance("gain")
    RES["adversarial_auc"] = float(roc_auc_score(ya, oof_a))
    RES["adversarial_auc_id_seul"] = float(roc_auc_score(ya, np.r_[train["id"], test["id"]]))
    imp = pd.Series(imp / imp.sum(), index=FEATS).sort_values(ascending=False)
    drift["adv_importance"] = imp
    log(f"   adversarial AUC = {RES['adversarial_auc']:.4f} ; AUC de l'id seul = {RES['adversarial_auc_id_seul']:.4f}")
    show("Importance adversarial (top 8)", imp.to_frame("part_gain"), 8)
    drift.to_csv(RAP / "audit_derive.csv")
    del Xa
except Exception:
    traceback.print_exc(); FAILED.append("4_derive")
    drift = pd.DataFrame(index=FEATS, columns=["psi", "adv_importance"]).astype(float)

# =====================================================================
# 5. DOUBLONS, CORRELATIONS, CODES, COLONNES DEDUITES, STRUCTURE
# =====================================================================
try:
    log("5a. doublons")
    h_tr = pd.util.hash_pandas_object(train[FEATS], index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(test[FEATS], index=False).to_numpy()
    RES["doublons_train_lignes_en_trop"] = int(pd.Series(h_tr).duplicated().sum())
    RES["doublons_test_lignes_en_trop"] = int(pd.Series(h_te).duplicated().sum())
    RES["lignes_test_identiques_a_une_ligne_train"] = int(np.isin(h_te, h_tr).sum())
    g = pd.DataFrame({"h": h_tr, "y": y}).groupby("h")["y"].agg(["size", "mean"])
    gd = g[g["size"] > 1]
    RES["doublons_groupes_train"] = int(len(gd))
    RES["doublons_groupes_cible_incoherente"] = int(((gd["mean"] > 0) & (gd["mean"] < 1)).sum())
    h2 = pd.util.hash_pandas_object(train[nocalc], index=False).to_numpy()
    h2t = pd.util.hash_pandas_object(test[nocalc], index=False).to_numpy()
    g2 = pd.DataFrame({"h": h2, "y": y}).groupby("h")["y"].agg(["size", "mean"])
    RES["doublons_hors_calc_lignes_en_trop_train"] = int(pd.Series(h2).duplicated().sum())
    RES["doublons_hors_calc_taille_max_groupe"] = int(g2["size"].max())
    RES["lignes_test_identiques_train_hors_calc"] = int(np.isin(h2t, h2).sum())
    print("Doublons:", {k: v for k, v in RES.items() if k.startswith("doublons") or k.startswith("lignes_test_ident")})

    log("5b. correlations")
    C = tr64.corr()
    Cn = tr64.replace(-1, np.nan).corr()
    pairs = []
    for i, a in enumerate(FEATS):
        for b in FEATS[i + 1:]:
            if max(abs(C.loc[a, b]), abs(Cn.loc[a, b] if not np.isnan(Cn.loc[a, b]) else 0)) > 0.3:
                pairs.append((a, b, C.loc[a, b], Cn.loc[a, b]))
    cp = pd.DataFrame(pairs, columns=["var1", "var2", "pearson_avec_-1", "pearson_-1_en_NaN"])
    cp["abs_max"] = cp[["pearson_avec_-1", "pearson_-1_en_NaN"]].abs().max(axis=1)
    cp = cp.sort_values("abs_max", ascending=False)
    cp.to_csv(RAP / "audit_correlations_fortes.csv", index=False)
    show("Paires |r|>0.3 (top 25)", cp, 25)
    RES["paires_corr>0.9"] = cp[cp.abs_max > 0.9][["var1", "var2", "abs_max"]].round(3).values.tolist()
    RES["paires_corr>0.5"] = cp[cp.abs_max > 0.5][["var1", "var2", "abs_max"]].round(3).values.tolist()
    cmax_calc = {c: float(max([abs(C.loc[c, o]) for o in FEATS if o != c])) for c in calc_only}
    RES["calc_corr_max_avec_autre_variable"] = max(cmax_calc.values())
    RES["calc_corr_max_avec_non_calc"] = float(max(abs(C.loc[c, o]) for c in calc_only for o in nocalc))
    print("Corr max calc vs non-calc:", RES["calc_corr_max_avec_non_calc"], "| calc vs autre calc/non-calc:", RES["calc_corr_max_avec_autre_variable"])
    # corr de chaque variable avec la cible (pour memoire)
    RES["corr_cible_top"] = tr64.corrwith(pd.Series(y, index=tr64.index)).abs().sort_values(ascending=False).head(8).round(4).to_dict()

    log("5c. valeurs, codes suspects, outliers")
    SPECIAL = [7, 9, 77, 99, 777, 999, 9999]
    val_rows, sus = [], []
    for c in FEATS:
        a = tr64[c]
        vc = a.value_counts()
        if len(vc) <= 25:
            gg = pd.DataFrame({"v": a, "y": y}).groupby("v")["y"].agg(["size", "mean"])
            for v, r in gg.iterrows():
                val_rows.append((c, v, int(r["size"]), r["size"] / N, r["mean"]))
        for sv in SPECIAL:
            if sv in vc.index and var_kind(c) == "num" or (sv in vc.index and var_kind(c) == "cat"):
                top = vc.index.max() == sv
                sus.append((c, sv, int(vc[sv]), vc[sv] / N, "plus grande valeur" if top else "au milieu"))
        neg = a[(a < 0) & (a != -1)]
        if len(neg):
            sus.append((c, "negatif_autre_que_-1", len(neg), len(neg) / N, ""))
    pd.DataFrame(val_rows, columns=["var", "valeur", "n", "part", "taux_cible"]).to_csv(RAP / "audit_valeurs.csv", index=False)
    show("Codes speciaux candidats (7,9,77,99,...)", pd.DataFrame(sus, columns=["var", "valeur", "n", "part", "position"]), 60)
    RES["codes_speciaux"] = [(s[0], s[1], s[2]) for s in sus]
    cont = [c for c in FEATS if var_kind(c) == "num" and tr64[c].nunique() > 20]
    q = tr64[cont].replace(-1, np.nan).quantile([0, .001, .01, .25, .5, .75, .99, .999, 1]).T
    q["ratio_max/p99.9"] = q[1.0] / q[0.999]
    show("Quantiles des variables continues (hors -1)", q)
    RES["quantiles_continues"] = q.round(4).to_dict("index")

    log("5d. colonnes deduites")
    # (i) groupes de binaires : distribution de la somme
    def pick(grp, nums, suf="_bin"): return [f"ps_{grp}_{n:02d}{suf}" for n in nums if f"ps_{grp}_{n:02d}{suf}" in FEATS]
    bin_sets = {"ind_06-09": pick("ind", [6, 7, 8, 9]), "ind_10-13": pick("ind", [10, 11, 12, 13]),
                "ind_16-18": pick("ind", [16, 17, 18]), "calc_15-20": pick("calc", range(15, 21))}
    RES["sommes_binaires"] = {}
    for k, cols in bin_sets.items():
        s = tr64[cols].sum(axis=1)
        vc = s.value_counts().sort_index()
        RES["sommes_binaires"][k] = {str(int(i)): int(n) for i, n in vc.items()}
        tcb = pd.Series(y).groupby(s.to_numpy()).mean().round(4).to_dict()
        print(f"Somme de {cols}: {RES['sommes_binaires'][k]} | taux cible par somme: {tcb}")
        # chaque binaire est-elle deduite des autres du groupe ?
        for c in cols:
            oth = [o for o in cols if o != c]
            code = np.zeros(N, dtype=np.int64)
            for o in oth: code = code * 2 + tr64[o].to_numpy().astype(np.int64)
            tab = pd.crosstab(code, tr64[c].to_numpy())
            RES.setdefault("binaire_deduite_des_autres_du_groupe", {})[c] = round(float(tab.max(axis=1).sum() / N), 5)
    print("Pureté de chaque binaire connaissant les autres du groupe:", RES["binaire_deduite_des_autres_du_groupe"])
    # (ii) determinisme entre variables discretes : B est-elle fonction de A ?
    low = [c for c in FEATS if np.unique(CODES[c]).size <= 120 and tr64[c].nunique() <= 120 and tr64[c].nunique() > 1]
    det = []
    for a in low:
        ca = CODES[a]; KA = ca.max() + 1
        for b in low:
            if a == b: continue
            cb = CODES[b]; KB = cb.max() + 1
            cnt = np.bincount(ca.astype(np.int64) * KB + cb, minlength=KA * KB).reshape(KA, KB)
            pur = cnt.max(axis=1).sum() / N
            base = np.bincount(cb).max() / N
            if pur >= 0.99 and (1 - pur) <= 0.1 * (1 - base) and KA > KB:
                det.append((a, b, KA, KB, pur, base))
    det = pd.DataFrame(det, columns=["A_determine", "B", "K_A", "K_B", "purete", "base_majoritaire_B"]).sort_values("purete", ascending=False)
    det.to_csv(RAP / "audit_dependances.csv", index=False)
    show("B quasi fonction de A (purete>=0.99, K_A>K_B)", det, 30)
    RES["dependances_fonctionnelles"] = det.head(20).round(4).values.tolist()

    log("5e. structure des variables continues")
    st = []
    for c in ["ps_reg_01", "ps_reg_02", "ps_reg_03", "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15", "ps_calc_01", "ps_calc_02", "ps_calc_03"]:
        a = tr64[c].to_numpy(); u = np.unique(a[a != -1])
        dec_ = next((d for d in range(0, 13) if np.allclose(np.round(u, d), u, atol=1e-9)), 13)
        d = dict(var=c, n_unique=len(u), min=u.min(), max=u.max(), decimales_max=dec_,
                 pas_min=float(np.diff(u).min()) if len(u) > 1 else np.nan, exemples=[round(float(x), 6) for x in u[:6]])
        for k in (1, 10, 100, 1000, 10000):
            x = u * u * k
            d[f"carre*{k}_entier"] = float((np.abs(x - np.round(x)) < 1e-3).mean())
        d["premier_k_ou_carre_entier"] = next((k for k in (1, 10, 100, 1000, 10000) if d[f"carre*{k}_entier"] > 0.99), None)
        for k in (10, 100):
            x = u * k; d[f"x*{k}_entier"] = float((np.abs(x - np.round(x)) < 1e-6).mean())
        st.append(d)
    st = pd.DataFrame(st).set_index("var")
    show("Structure ps_reg/car continues", st)
    RES["structure_continues"] = st.drop(columns=["exemples"]).round(4).to_dict("index")
    RES["exemples_valeurs"] = st["exemples"].to_dict()
    print("ps_car_15 valeurs distinctes:", np.unique(tr64["ps_car_15"]).tolist())
    # ps_car_13 vs ps_car_12 et ps_reg_03 vs reg_01/02
    for (u_, v_) in [("ps_car_13", "ps_car_12"), ("ps_reg_03", "ps_reg_02"), ("ps_reg_03", "ps_reg_01"), ("ps_car_14", "ps_car_12")]:
        m_ = (tr64[u_] != -1) & (tr64[v_] != -1)
        print(f"corr({u_},{v_}) hors -1 = {tr64.loc[m_, u_].corr(tr64.loc[m_, v_]):.3f}")
except Exception:
    traceback.print_exc(); FAILED.append("5_structure")

# =====================================================================
# 6. FIGURES
# =====================================================================
log("6. figures")
BLUE, RED, GREY, ORANGE = "#2a6f97", "#d1495b", "#8d99ae", "#edae49"
GCOL = {"ind": BLUE, "reg": ORANGE, "car": RED, "calc": GREY}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
try:
    # 6.1 cible
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cnt = pd.Series(y).value_counts().sort_index()
    ax[0].bar(["0 (pas de sinistre)", "1 (sinistre)"], cnt.values, color=[GREY, RED])
    for i, v in enumerate(cnt.values): ax[0].text(i, v, f"{v:,}\n({100*v/N:.2f} %)".replace(",", " "), ha="center", va="bottom")
    ax[0].set_title("Distribution de la cible (train)"); ax[0].set_ylim(0, cnt.max() * 1.18)
    rate = pd.Series(y).groupby(dec.to_numpy()).agg(["mean", "size"])
    ci = 1.96 * np.sqrt(rate["mean"] * (1 - rate["mean"]) / rate["size"])
    ax[1].errorbar(range(1, 11), rate["mean"] * 100, yerr=ci * 100, fmt="o-", color=BLUE, capsize=3)
    ax[1].axhline(BASE * 100, color=RED, ls="--", lw=1, label="taux global")
    ax[1].set_xlabel("décile de l'identifiant (id)"); ax[1].set_ylabel("taux de sinistre (%)")
    ax[1].set_title("Taux de sinistre selon l'id (contrôle de fuite)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "01_cible.png"); plt.close(fig)

    # 6.2 manquants
    pm = prof[prof.n_manquants_train > 0].sort_values("pct_manq_train")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pm) + 1.5))
    yy_ = np.arange(len(pm))
    ax.barh(yy_ - 0.2, pm.pct_manq_train, 0.4, color=BLUE, label="train")
    ax.barh(yy_ + 0.2, pm.pct_manq_test, 0.4, color=ORANGE, label="test")
    ax.set_yticks(yy_); ax.set_yticklabels(pm.index); ax.set_xlabel("% de valeurs manquantes (-1)")
    ax.set_title("Valeurs manquantes par variable (les autres variables n'en ont aucune)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "01_manquants.png"); plt.close(fig)

    # 6.3 top 20 signal
    t20 = sig.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    yy_ = np.arange(len(t20))
    ax.barh(yy_ + 0.2, t20.abs_gini_brut, 0.4, color=BLUE, label="|Gini| brut (-1 tel quel)")
    ax.barh(yy_ - 0.2, t20.gini_te.clip(lower=0), 0.4, color=ORANGE, label="Gini encodage cible (CV 5 folds)")
    ax.axvline(thr_fam, color=RED, ls="--", lw=1, label=f"seuil de bruit ({thr_fam:.3f})")
    ax.set_yticks(yy_); ax.set_yticklabels(t20.index); ax.set_xlabel("Gini univarié")
    ax.set_title("Top 20 des variables par signal univarié"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "01_gini_top20.png"); plt.close(fig)

    # 6.4 signal par groupe
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, gname in enumerate(["ind", "reg", "car", "calc"]):
        s = sig[sig.groupe == gname]
        xj = i + np.random.default_rng(1).uniform(-0.18, 0.18, len(s))
        ax.scatter(xj, s.signal, color=GCOL[gname], s=28, alpha=0.85)
    ax.axhline(thr_fam, color=RED, ls="--", lw=1, label=f"bruit (seuil 5 % sur 57 variables) = {thr_fam:.3f}")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{g}\n({(sig.groupe==g).sum()} var.)" for g in ["ind", "reg", "car", "calc"]])
    ax.set_ylabel("meilleur Gini univarié"); ax.set_title("Signal univarié par groupe de variables"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "01_gini_par_groupe.png"); plt.close(fig)

    # 6.5 manquant vs cible
    pm2 = prof[prof.n_manquants_train >= 100].sort_values("pct_manq_train", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xx = np.arange(len(pm2)); w = 0.38
    for off, col, lab, key, nn in [(-w / 2, RED, "manquant (-1)", "taux_cible_manquant", pm2.n_manquants_train),
                                    (w / 2, BLUE, "renseigné", "taux_cible_present", N - pm2.n_manquants_train)]:
        p = pm2[key]; e = 1.96 * np.sqrt(p * (1 - p) / nn)
        ax.bar(xx + off, p * 100, w, yerr=e * 100, color=col, label=lab, capsize=3)
    ax.axhline(BASE * 100, color="k", ls="--", lw=1, label="taux global")
    ax.set_xticks(xx); ax.set_xticklabels([f"{v}\n({p:.0f} % manq.)" for v, p in zip(pm2.index, pm2.pct_manq_train)], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("taux de sinistre (%)"); ax.set_title("Taux de sinistre : valeur manquante vs renseignée (IC 95 %)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "01_manquants_vs_cible.png"); plt.close(fig)

    # 6.6 correlations (hors calc)
    ordre = nocalc
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(C.loc[ordre, ordre].to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordre))); ax.set_xticklabels(ordre, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ordre))); ax.set_yticklabels(ordre, fontsize=6)
    fig.colorbar(im, shrink=0.8, label="corrélation de Pearson")
    ax.set_title(f"Corrélations hors ps_calc (corr. max des ps_calc avec toute variable : {RES['calc_corr_max_avec_autre_variable']:.3f})")
    fig.tight_layout(); fig.savefig(FIG / "01_correlations.png"); plt.close(fig)

    # 6.7 derive
    d1 = drift.sort_values("psi", ascending=False).head(12).iloc[::-1]
    d2 = drift["adv_importance"].dropna().sort_values(ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].barh(d1.index, d1.psi, color=BLUE); ax[0].axvline(0.1, color=RED, ls="--", lw=1, label="seuil 0,1 (dérive notable)")
    ax[0].set_title("PSI train/test (plus grandes valeurs)"); ax[0].legend()
    ax[1].barh(d2.index, d2.values * 100, color=ORANGE)
    ax[1].set_title(f"Adversarial validation : AUC = {RES.get('adversarial_auc', float('nan')):.3f}\nimportance (% du gain)")
    fig.tight_layout(); fig.savefig(FIG / "01_derive.png"); plt.close(fig)
except Exception:
    traceback.print_exc(); FAILED.append("6_figures")

# =====================================================================
# 7. SAUVEGARDES
# =====================================================================
full = prof.join(sig.drop(columns=["groupe", "type"]), how="left").join(drift[[c for c in ["psi", "ks", "adv_importance"] if c in drift]], how="left")
full.sort_values("signal", ascending=False).to_csv(RAP / "audit_variables.csv")
with open(RAP / "audit_resume.json", "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log(f"FIN. Sections en erreur: {FAILED or 'aucune'}")
raise SystemExit(1 if FAILED else 0)
