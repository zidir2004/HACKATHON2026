"""05_analyse.py — analyse métier et explicabilité de la baseline E01 (LightGBM sur variables brutes).

Usage (depuis le dossier Hackathon) :  python -u usine/src/05_analyse.py

Étapes :
  1. Réentraîne le modèle E01 (mêmes paramètres) en gardant le fold 0 pour la validation
  2. SHAP (TreeExplainer) sur 20 000 lignes de validation : importance, beeswarm, dépendances, groupes
  3. Pouvoir de discrimination sur l'OOF E01 : ROC, courbe de gains, déciles de score
  4. Profils : 10 % les plus risqués vs 10 % les moins risqués
  5. Taux de sinistre par modalité des variables catégorielles (effectifs + intervalle de confiance)
Sorties : usine/figures/05_*.png, usine/rapports/05_*.csv, usine/rapports/05_resultats.json
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent))
import utils as U  # noqa: E402
T = importlib.import_module("03_train")      # on réutilise la config E01 (mêmes paramètres)
import lightgbm as lgb  # noqa: E402

FIG, RAP = U.USINE / "figures", U.USINE / "rapports"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
BLEU, ORANGE, GRIS = "#1f5fa8", "#e07b00", "#8a8a8a"
RES: dict = {}


def groupe(nom: str) -> str:
    """Famille d'une variable d'après son préfixe (ind / reg / car / calc)."""
    return nom.split("_")[1]


def sauver(nom: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / nom, bbox_inches="tight")
    plt.close()
    print("  figure", nom, flush=True)


# ---------------------------------------------------------------------------
# 0. Données + réentraînement du modèle E01 sur les folds 1-4, validation = fold 0
# ---------------------------------------------------------------------------
train, test = U.charger_donnees("v0")
cfg = T.CONFIGS["E01"]
feats = U.liste_features(train, cfg["exclure"])
X, y = train[feats].astype(np.float32), train[U.TARGET].to_numpy()
fold = U.charger_folds(train)
tr, va = fold != 0, fold == 0
model = lgb.train({**cfg["params"], "n_jobs": 8}, lgb.Dataset(X[tr], y[tr]), num_boost_round=cfg["n_max"],
                  valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(cfg["early_stop"], verbose=False)])
print(f"modèle fold 0 : {model.best_iteration} arbres, gini={U.gini(y[va], model.predict(X[va], num_iteration=model.best_iteration)):.5f}")

# ---------------------------------------------------------------------------
# 1. SHAP sur 20 000 lignes de validation
# ---------------------------------------------------------------------------
Xv = X[va].sample(20000, random_state=0)
expl = shap.TreeExplainer(model)
sv = expl.shap_values(Xv, check_additivity=False)
if isinstance(sv, list):       # anciennes versions de shap : liste [classe0, classe1]
    sv = sv[1]
sv = np.asarray(sv)
imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
imp.rename("shap_abs_moyen").to_csv(RAP / "05_shap_importance.csv")
RES["top10_shap"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}

# 1a. barres, top 20
top = imp.head(20)[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top.index, top.values, color=[{"ind": BLEU, "reg": "#2a9d8f", "car": ORANGE, "calc": GRIS}[groupe(v)] for v in top.index])
plt.xlabel("Importance moyenne |SHAP| (effet moyen sur le log-odds du risque)")
plt.title("Les 20 variables qui pèsent le plus dans le modèle (E01)")
for g, c in [("ind (conducteur)", BLEU), ("reg (région)", "#2a9d8f"), ("car (véhicule)", ORANGE), ("calc (calculée)", GRIS)]:
    plt.barh([], [], color=c, label=g)
plt.legend(loc="lower right", title="Famille")
sauver("05_shap_importance.png")

# 1b. beeswarm
plt.figure(figsize=(9, 8))
shap.summary_plot(sv, Xv, max_display=15, show=False, plot_size=None)
plt.title("Effet de chaque variable sur le risque, assuré par assuré\n(rouge = valeur élevée ; à droite = risque augmenté)")
plt.xlabel("Valeur SHAP (impact sur le log-odds du sinistre)")
sauver("05_shap_beeswarm.png")

# 1c. dépendances des 4 variables principales
for v in imp.index[:4]:
    x, s = Xv[v].to_numpy(), sv[:, feats.index(v)]
    plt.figure(figsize=(8, 5))
    plt.scatter(x, s, s=5, alpha=0.25, color=BLEU)
    d = pd.DataFrame({"x": x, "s": s})
    if d["x"].nunique() > 25:
        d["b"] = pd.qcut(d["x"], 25, duplicates="drop")
        m = d.groupby("b", observed=True).agg(x=("x", "mean"), s=("s", "mean"))
    else:
        m = d.groupby("x").agg(x=("x", "mean"), s=("s", "mean"))
    plt.plot(m["x"], m["s"], color=ORANGE, lw=2.5, label="effet moyen")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel(f"Valeur de {v}  (-1 = manquant)")
    plt.ylabel("Valeur SHAP (impact sur le risque)")
    plt.title(f"Comment {v} modifie la prédiction")
    plt.legend()
    sauver(f"05_shap_dep_{v}.png")

# 1d. importance par famille
g = imp.groupby(imp.index.map(groupe)).sum()
nb = pd.Series([groupe(f) for f in feats]).value_counts()
part = (g / g.sum() * 100).sort_values()
RES["part_groupes_pct"] = {k: round(float(v), 1) for k, v in part.items()}
RES["nb_vars_groupes"] = {k: int(v) for k, v in nb.items()}
noms = {"ind": "ind (conducteur)", "reg": "reg (région)", "car": "car (véhicule)", "calc": "calc (calculées)"}
plt.figure(figsize=(8, 4.5))
cols = [GRIS if k == "calc" else BLEU for k in part.index]
plt.barh([f"{noms[k]} — {nb[k]} variables" for k in part.index], part.values, color=cols)
for i, v in enumerate(part.values):
    plt.text(v + 0.5, i, f"{v:.1f} %", va="center")
plt.xlabel("Part de l'importance totale |SHAP| (%)")
plt.title("Les 20 variables 'calc' pèsent peu dans le modèle")
sauver("05_importance_groupes.png")

# ---------------------------------------------------------------------------
# 2. Pouvoir de discrimination sur l'OOF E01
# ---------------------------------------------------------------------------
oof = pd.read_csv(U.OOF_DIR / "E01_oof.csv")
yo, po = oof["target"].to_numpy(), oof["oof"].to_numpy()
fpr, tpr, _ = roc_curve(yo, po)
gini_oof = U.gini(yo, po)
RES["gini_oof"], RES["auc_oof"] = round(gini_oof, 5), round((gini_oof + 1) / 2, 5)

plt.figure(figsize=(6.5, 6))
plt.plot(fpr, tpr, color=BLEU, lw=2.5, label=f"Modèle E01 (AUC = {(gini_oof+1)/2:.3f} ; Gini = {gini_oof:.3f})")
plt.plot([0, 1], [0, 1], "--", color=GRIS, label="Hasard (AUC = 0,5)")
plt.xlabel("Faux positifs : part des non-sinistrés classés à risque")
plt.ylabel("Vrais positifs : part des sinistrés détectés")
plt.title("Courbe ROC (prédictions hors-échantillon, 5 folds)")
plt.legend(loc="lower right")
sauver("05_roc.png")

# courbe de gains : on trie du plus risqué au moins risqué
o = np.argsort(-po)
cum = np.cumsum(yo[o]) / yo.sum()
pop = np.arange(1, len(yo) + 1) / len(yo)
gains = {f"top{int(q*100)}": round(float(cum[int(q * len(yo)) - 1]) * 100, 1) for q in (0.05, 0.10, 0.20, 0.30, 0.50)}
RES["gains_pct_sinistres"] = gains
plt.figure(figsize=(7, 6))
plt.plot(pop * 100, cum * 100, color=BLEU, lw=2.5, label="Modèle E01")
plt.plot([0, 100], [0, 100], "--", color=GRIS, label="Sans modèle (hasard)")
for q in (0.2, 0.5):
    v = cum[int(q * len(yo)) - 1] * 100
    plt.scatter([q * 100], [v], color=ORANGE, zorder=5)
    plt.annotate(f"{q*100:.0f} % les plus risqués\n= {v:.0f} % des sinistres", (q * 100, v), xytext=(q * 100 + 6, v - 14),
                 arrowprops=dict(arrowstyle="->", color=ORANGE))
plt.xlabel("% des assurés, classés du plus au moins risqué selon le modèle")
plt.ylabel("% des sinistres capturés")
plt.title("Courbe de gains (Lorenz) : le modèle concentre les sinistres")
plt.legend(loc="lower right")
sauver("05_gains.png")

# déciles de score (décile 1 = le moins risqué, 10 = le plus risqué)
oof["decile"] = pd.qcut(oof["oof"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec = oof.groupby("decile").agg(taux=("target", "mean"), n=("target", "size"), sinistres=("target", "sum"))
dec.to_csv(RAP / "05_deciles.csv")
ratio = dec.loc[10, "taux"] / dec.loc[1, "taux"]
RES["deciles_taux_pct"] = {int(k): round(float(v) * 100, 2) for k, v in dec["taux"].items()}
RES["ratio_d10_d1"] = round(float(ratio), 2)
plt.figure(figsize=(9, 5))
plt.bar(dec.index, dec["taux"] * 100, color=[ORANGE if d == 10 else BLEU for d in dec.index])
plt.axhline(yo.mean() * 100, color="k", ls="--", lw=1, label=f"Taux moyen {yo.mean()*100:.2f} %")
for d, v in dec["taux"].items():
    plt.text(d, v * 100 + 0.05, f"{v*100:.1f}", ha="center", fontsize=9)
plt.xticks(range(1, 11))
plt.xlabel("Décile de score (1 = 10 % les moins risqués, 10 = 10 % les plus risqués)")
plt.ylabel("Taux de sinistre observé (%)")
plt.title(f"Taux de sinistre par décile de score : le décile 10 est {ratio:.1f} fois plus risqué que le décile 1")
plt.legend()
sauver("05_deciles.png")

# ---------------------------------------------------------------------------
# 3. Profils : 10 % les plus risqués vs 10 % les moins risqués (variables principales)
# ---------------------------------------------------------------------------
full = train.merge(oof[[U.ID, "decile"]], on=U.ID)
haut, bas = full[full["decile"] == 10], full[full["decile"] == 1]
rows = []
for v in imp.head(15).index:
    sd = full[v].std()
    rows.append(dict(variable=v, moy_risque_haut=haut[v].mean(), moy_risque_bas=bas[v].mean(), moy_globale=full[v].mean(),
                     ecart_standardise=(haut[v].mean() - bas[v].mean()) / sd,
                     pct_manquant_haut=(haut[v] == -1).mean() * 100, pct_manquant_bas=(bas[v] == -1).mean() * 100,
                     modalite_dominante_haut=haut[v].mode().iloc[0], modalite_dominante_bas=bas[v].mode().iloc[0]))
prof = pd.DataFrame(rows)
prof.to_csv(RAP / "05_profils.csv", index=False)
RES["profils"] = prof.round(3).to_dict("records")
p = prof.sort_values("ecart_standardise")
plt.figure(figsize=(9, 7))
plt.barh(p["variable"], p["ecart_standardise"], color=[ORANGE if e > 0 else BLEU for e in p["ecart_standardise"]])
plt.axvline(0, color="k", lw=0.8)
plt.xlabel("Écart de moyenne en écarts-types (10 % les plus risqués moins 10 % les moins risqués)")
plt.title("Profil des assurés les plus risqués selon le modèle (15 variables principales)")
sauver("05_profils.png")

# ---------------------------------------------------------------------------
# 4. Taux de sinistre par modalité des variables catégorielles
# ---------------------------------------------------------------------------
taux_global = y.mean()
cats = [c for c in imp.index if c.endswith("_cat")]
choisies = ["ps_car_11_cat"] + [c for c in cats if c != "ps_car_11_cat"][:3]
RES["cat_choisies"] = choisies


def table_modalites(v: str) -> pd.DataFrame:
    t = train.groupby(v)[U.TARGET].agg(taux="mean", n="size").reset_index()
    t["ic"] = 1.96 * np.sqrt(t["taux"] * (1 - t["taux"]) / t["n"])
    return t


t11 = table_modalites("ps_car_11_cat")
t11.to_csv(RAP / "05_modalites_ps_car_11_cat.csv", index=False)
RES["car_11_cat"] = dict(nb_modalites=int(len(t11)), nb_moins_500=int((t11.n < 500).sum()), nb_moins_100=int((t11.n < 100).sum()),
                         min_taux_pct=round(float(t11.taux.min() * 100), 2), max_taux_pct=round(float(t11.taux.max() * 100), 2),
                         part_pop_modalites_moins_500_pct=round(float(t11[t11.n < 500].n.sum() / t11.n.sum() * 100), 1),
                         ic_median_pt=round(float(t11.ic.median() * 100), 2))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].errorbar(t11["n"], t11["taux"] * 100, yerr=t11["ic"] * 100, fmt="o", color=BLEU, alpha=0.7, ecolor="#9db8d9", ms=4)
ax[0].axhline(taux_global * 100, color="k", ls="--", label=f"Taux moyen {taux_global*100:.2f} %")
ax[0].set_xscale("log")
ax[0].set_xlabel("Nombre d'assurés dans la modalité (échelle log)")
ax[0].set_ylabel("Taux de sinistre (%) ± intervalle à 95 %")
ax[0].set_title("Effectifs de 851 à 63 780 assurés : intervalle de confiance large pour les petites modalités")
ax[0].legend()
tt = t11.sort_values("taux")
ax[1].bar(range(len(tt)), tt["taux"] * 100, color=[ORANGE if n < 3000 else BLEU for n in tt["n"]])
ax[1].axhline(taux_global * 100, color="k", ls="--")
ax[1].set_xlabel("Les 104 modalités triées par taux (orange = moins de 3 000 assurés)")
ax[1].set_ylabel("Taux de sinistre (%)")
ax[1].set_title("ps_car_11_cat : des taux de 2 % à 7,4 % selon la modalité")
fig.suptitle("Variable ps_car_11_cat (104 modalités) : le taux varie du simple au triple, sans ordre naturel")
sauver("05_modalites_ps_car_11_cat.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for a, v in zip(axes, choisies[1:]):
    t = table_modalites(v)
    t.to_csv(RAP / f"05_modalites_{v}.csv", index=False)
    a.bar(t[v].astype(int).astype(str), t["taux"] * 100, yerr=t["ic"] * 100, color=BLEU, capsize=3, ecolor=GRIS)
    for i, (n, tx) in enumerate(zip(t["n"], t["taux"])):
        a.text(i, 0.1, f"n={n}", rotation=90, ha="center", va="bottom", color="white", fontsize=8)
    a.axhline(taux_global * 100, color="k", ls="--", lw=1)
    a.set_title(v)
    a.set_xlabel("Modalité (-1 = manquant)")
    a.set_ylabel("Taux de sinistre (%)")
fig.suptitle("Taux de sinistre par modalité pour les autres variables catégorielles importantes (tirets = taux moyen)")
sauver("05_modalites_autres_cat.png")

(RAP / "05_resultats.json").write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in RES.items() if k != "profils"}, indent=1, ensure_ascii=False, default=float))
print("TERMINE")
