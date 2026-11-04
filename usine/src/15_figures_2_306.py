"""15_figures_2_306.py — figures de soutenance pour la soumission finale 2_306 (modèle E33f).

Usage (depuis le dossier Hackathon) :  python -u usine/src/15_figures_2_306.py
Sorties : usine/figures/11_*.png, usine/rapports/15_resultats.json
Lit seulement experiences/registre.csv, comparaisons.csv et les OOF ; pandas + numpy + matplotlib, CPU, ~10 s.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[2]
USINE = RACINE / "usine"
FIG, RAP, EXP = USINE / "figures", USINE / "rapports", USINE / "experiences"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 12, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 14, "axes.titleweight": "bold"})
BLEU, ORANGE, GRIS, ROUGE, VERT = "#1f5fa8", "#e07b00", "#8a8a8a", "#c0392b", "#2e8b57"
RES: dict = {}


def fr(x, nd=3):
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{nd}f}".replace(".", ",")


FMT = lambda nd: FuncFormatter(lambda v, _: fr(v, nd))


def sauver(nom):
    plt.tight_layout()
    plt.savefig(FIG / nom, bbox_inches="tight")
    plt.close()
    print("  figure", nom, flush=True)


reg = pd.read_csv(EXP / "registre.csv").drop_duplicates("id", keep="last").set_index("id")
cmp_ = pd.read_csv(EXP / "comparaisons.csv").drop_duplicates(["exp", "ref"], keep="last")
cv = lambda e: (float(reg.loc[e, "score_cv_moyen"]), float(reg.loc[e, "score_cv_std"]))

# ---------------------------------------------------------------- 1. progression des soumissions
sub = [("2_302", "E01", "LightGBM\nbrut"), ("2_303", "E09", "mélange de\n3 boostings"),
       ("2_305", "E34", "mélange,\n3 graines LGBM"), ("2_306", "E33f", "mélange de 4\n(+ réseau DAE)"),
       ("2_307", "E56", "mélange\n(2 réseaux)")]
serveur = {"2_306": 0.30079, "2_307": 0.3003}
fig, ax = plt.subplots(figsize=(12, 6.5))
x = np.arange(len(sub))
m = [cv(e)[0] for _, e, _ in sub]
s = [cv(e)[1] for _, e, _ in sub]
cols = [VERT if n == "2_306" else BLEU for n, _, _ in sub]
ax.errorbar(x, m, yerr=s, fmt="none", ecolor="#444", capsize=6, lw=1.8, zorder=2)
ax.scatter(x, m, s=130, c=cols, zorder=3, label="Gini en validation croisée (moyenne ± écart-type entre plis)")
for i, v in enumerate(m):
    ax.annotate(fr(v, 4), (x[i], v), textcoords="offset points", xytext=(30, -22), ha="left", fontsize=11, color="#222")
for n, v in serveur.items():
    i = [a for a, _, _ in sub].index(n)
    ax.scatter(i, v, marker="D", s=150, c=ORANGE, zorder=4, edgecolor="k",
               label="Score officiel du serveur" if n == "2_306" else None)
    ax.annotate(fr(v, 4), (i, v), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=11, fontweight="bold", color=ORANGE)
ax.axhline(0.23923, color=ROUGE, ls="--", lw=1.6)
ax.text(len(sub) - 0.55, 0.2415, "baseline logistique : 0,239", color=ROUGE, ha="right", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n({e})\n{d}" for n, e, d in sub])
ax.set_ylim(0.225, 0.318)
ax.yaxis.set_major_formatter(FMT(2))
ax.set_ylabel("Gini (2 × AUC − 1)")
ax.set_title("Progression des soumissions : le score serveur dépasse la validation croisée")
ax.text(0.01, 0.97, "Écart serveur / validation (~ +0,015) : de l'ordre de 2 écarts-types entre plis (0,007),\n"
        "variance d'échantillonnage du jeu de test, pas une fuite (test d'adversaire : AUC 0,50).",
        transform=ax.transAxes, va="top", fontsize=10.5, color="#333")
ax.legend(loc="center", fontsize=10.5, frameon=True, bbox_to_anchor=(0.5, 0.3))
sauver("11_progression_soumissions.png")

# ---------------------------------------------------------------- 2. schéma du modèle final
mem = {"E30": "LightGBM\n(paramètres 1re place\nKaggle + one-hot des _cat)", "E07": "XGBoost\n(GPU)", "E08": "CatBoost\n(GPU)",
       "E32c": "Réseau 4500-1000-1000\nsur autoencodeur débruiteur\n(moyenne de 3 graines)"}
fig, ax = plt.subplots(figsize=(15, 7.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 7.2); ax.axis("off"); ax.grid(False)


def boite(xc, yc, w, h, txt, fc, ec="#333", fs=11.5, bold=False):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0.04,rounding_size=0.15", fc=fc, ec=ec, lw=1.5))
    ax.text(xc, yc, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def fleche(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#444"))


boite(1.3, 3.6, 2.1, 1.7, "Données brutes\n446 409 assurés\n57 variables", "#e8eef7", bold=True)
boite(4.1, 3.6, 2.5, 2.2, "Version v1\n- retrait des 20 ps_calc\n  (bruit)\n- -1 gardé comme\n  modalité (manquant)\n- one-hot des _cat\n  (modèles E30, réseau)", "#eaf5ee")
fleche(2.4, 3.6, 2.85, 3.6)
ys = [6.0, 4.5, 3.0, 1.35]
for (k, t), y in zip(mem.items(), ys):
    g = cv(k)[0]
    boite(7.55, y, 3.3, 1.25, f"{k} : {t}\nGini seul {fr(g, 4)}", "#fdf1e0" if k != "E32c" else "#f6e6f2", fs=10.2)
    fleche(5.4, 3.6, 5.9, y)
    fleche(9.2, y, 10.0, 3.6)
boite(11.0, 3.6, 1.9, 1.7, "Moyenne des\nrangs\n(poids égaux)", "#dbe7f7", bold=True)
fleche(11.95, 3.6, 12.3, 3.6)
boite(13.4, 3.6, 2.0, 1.9, "Remise à\nl'échelle en\nprobabilité\n(taux 3,65 %)", "#dbe7f7", bold=True)
boite(13.4, 1.0, 2.2, 0.8, "2_306.csv", "#d6ecd9", ec=VERT, bold=True, fs=13)
fleche(13.4, 2.6, 13.4, 1.45)
ax.text(11.0, 1.0, "Gini CV du mélange\n" + fr(cv("E33f")[0], 4) + f" ± {fr(cv('E33f')[1], 4)}\nServeur : 0,3008",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=VERT)
ax.text(7.5, 7.0, "Pipeline du modèle final (soumission 2_306, expérience E33f)", ha="center", fontsize=15, fontweight="bold")
ax.text(7.55, 0.2, "Même validation croisée (5 plis figés) pour tous les membres ; le réseau est le plus faible seul mais le plus différent.",
        ha="center", fontsize=10.5, color="#444")
sauver("11_schema_modele_final.png")

# ---------------------------------------------------------------- OOF
oof = {}
for e in ["E30", "E07", "E08", "E32c", "E33f"]:
    d = pd.read_csv(EXP / "oof" / f"{e}_oof.csv").set_index("id")
    oof[e] = d
y = oof["E33f"]["target"]
O = pd.DataFrame({e: oof[e]["oof"].reindex(oof["E33f"].index) for e in oof})

# ---------------------------------------------------------------- 3. corrélations de rang
mem4 = ["E30", "E07", "E08", "E32c"]
C = O[mem4].rank().corr()
noms = {"E30": "E30\nLightGBM", "E07": "E07\nXGBoost", "E08": "E08\nCatBoost", "E32c": "E32c\nRéseau + DAE"}
fig, ax = plt.subplots(figsize=(8.5, 7.2))
im = ax.imshow(C.values, cmap="Blues", vmin=0.8, vmax=1.0)
ax.grid(False)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([noms[e] for e in mem4]); ax.set_yticklabels([noms[e] for e in mem4])
for i in range(4):
    for j in range(4):
        v = C.values[i, j]
        ax.text(j, i, fr(v, 3), ha="center", va="center", fontsize=14, fontweight="bold", color="white" if v > 0.93 else "#111")
ax.set_title("Corrélation de rang des prédictions (OOF) entre les 4 membres")
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.ax.yaxis.set_major_formatter(FMT(2))
ax.text(0.5, -0.17, "Entre arbres : 0,94 à 0,96. Réseau vs arbres : 0,91 à 0,92 -> il fait des erreurs différentes.",
        transform=ax.transAxes, ha="center", fontsize=11)
sauver("11_correlations_membres.png")
RES["corr_rang"] = {f"{a}-{b}": round(float(C.loc[a, b]), 4) for i, a in enumerate(mem4) for b in mem4[i + 1:]}

# ---------------------------------------------------------------- 4. gains et déciles
d = pd.DataFrame({"y": y, "s": O["E33f"]})
d = d.sort_values("s", ascending=False, kind="mergesort").reset_index(drop=True)
n, P = len(d), d["y"].sum()
cumpos = d["y"].cumsum().to_numpy() / P
part = np.arange(1, n + 1) / n
top20 = float(cumpos[int(0.2 * n) - 1]); top10 = float(cumpos[int(0.1 * n) - 1]); top50 = float(cumpos[int(0.5 * n) - 1])
rk = O["E33f"].rank(method="first")
dec = pd.qcut(rk, 10, labels=False) + 1  # 10 = plus risqué
tab = pd.DataFrame({"y": y, "dec": dec}).groupby("dec")["y"].agg(["mean", "sum", "size"])
ratio = float(tab.loc[10, "mean"] / tab.loc[1, "mean"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
a1.plot(part, cumpos, color=BLEU, lw=2.6, label="Modèle E33f")
a1.plot([0, 1], [0, 1], color=GRIS, ls="--", label="Hasard")
a1.plot([0.2, 0.2], [0, top20], color=ORANGE, ls=":"); a1.plot([0, 0.2], [top20, top20], color=ORANGE, ls=":")
a1.scatter([0.2], [top20], color=ORANGE, s=80, zorder=5)
a1.annotate(f"Les 20 % les plus risqués\n= {fr(100 * top20, 1)} % des sinistres", (0.2, top20), xytext=(0.33, top20 - 0.17),
            fontsize=12.5, fontweight="bold", color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))} %"))
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v * 100))} %"))
a1.set_xlabel("Part des assurés, du plus au moins risqué (selon le modèle)"); a1.set_ylabel("Part des sinistres capturée")
a1.set_title("Courbe de gains (validation croisée)"); a1.legend(loc="lower right", frameon=False)
taux = tab["mean"].to_numpy() * 100
a2.bar(tab.index, taux, color=[BLEU] * 9 + [ROUGE], edgecolor="white")
a2.axhline(100 * y.mean(), color=GRIS, ls="--"); a2.text(1.0, 100 * y.mean() + 0.12, f"moyenne {fr(100 * y.mean(), 2)} %", color="#555", fontsize=10.5)
for i, v in zip(tab.index, taux):
    a2.text(i, v + 0.1, fr(v, 1), ha="center", fontsize=10.5)
a2.set_xticks(range(1, 11)); a2.set_xticklabels(["1\n(moins\nrisqué)"] + [str(i) for i in range(2, 10)] + ["10\n(plus\nrisqué)"])
a2.set_ylabel("Taux de sinistre (%)"); a2.set_xlabel("Décile de risque prédit")
a2.set_title(f"Taux de sinistre par décile : ×{fr(ratio, 1)} entre le décile 10 et le 1")
a2.yaxis.set_major_formatter(FMT(0))
fig.suptitle("Utilité métier du modèle final E33f (prédictions hors échantillon)", fontsize=15, fontweight="bold", y=1.02)
sauver("11_gains_deciles_E33f.png")
RES.update(top10=top10, top20=top20, top50=top50, taux_d1=float(taux[0]), taux_d10=float(taux[-1]), ratio_d10_d1=ratio,
           taux_moyen=float(y.mean()), n=int(n))
tab.assign(taux=tab["mean"]).to_csv(RAP / "15_deciles_E33f.csv")

# ---------------------------------------------------------------- 5. ce qui a marché
choix = [("E02", "E01", "retrait des 20 ps_calc"), ("E03", "E02", "indicateurs de manquants"),
         ("E04", "E02", "sans ps_ind_14, one-hot regroupés"), ("E05", "E02", "encodage cible de ps_car_11_cat"),
         ("E06", "E02", "catégorielles natives LightGBM"), ("E09", "E02", "mélange LGBM + XGB + CatBoost"),
         ("E13", "E02", "taux d'apprentissage 0,02 -> 0,01"), ("E17", "E09", "3 graines de LightGBM dans le mélange"),
         ("E19", "E02", "12 variables construites d'un coup"), ("E28", "E02m", "somme des binaires (E27), 3 graines"),
         ("E29", "E13", "paramètres bruts de Jahrer"), ("E30b", "E29", "+ one-hot des _cat"),
         ("E34", "E17", "mélange avec LightGBM Jahrer (3 graines)"), ("E33f", "E09", "mélange à 4 avec le réseau DAE"),
         ("E41", "E33f", "mélange avec DAE repris (E40c)"), ("E52", "E52_g0", "MLP sans DAE, 3 graines")]
lig = []
for e, r, t in choix:
    row = cmp_[(cmp_.exp == e) & (cmp_.ref == r)].iloc[0]
    lig.append((e, r, t, row.d_moy, row.d_std / np.sqrt(5), row.verdict, row.folds_gagnes))
df = pd.DataFrame(lig, columns=["exp", "ref", "texte", "d", "se", "verdict", "plis"])
df.to_csv(RAP / "15_ce_qui_a_marche.csv", index=False)
fig, ax = plt.subplots(figsize=(13, 8.5))
yy = np.arange(len(df))[::-1]
cmap = {"garde": VERT, "rejete": ROUGE, "neutre": GRIS}
ax.barh(yy, df.d, xerr=df.se, color=[cmap[v] for v in df.verdict], ecolor="#222", capsize=4, height=0.65)
ax.axvline(0, color="k", lw=1)
ax.set_yticks(yy)
ax.set_yticklabels([f"{r.exp} : {r.texte} (vs {r.ref})" for r in df.itertuples()], fontsize=10.5)
for yi, r in zip(yy, df.itertuples()):
    off = 0.0004
    ax.text(r.d + (r.se + off if r.d >= 0 else -(r.se + off)), yi, f"{'+' if r.d >= 0 else '−'}{fr(abs(r.d), 4)}  ({r.plis})",
            va="center", ha="left" if r.d >= 0 else "right", fontsize=9.5)
ax.set_xlim(-0.0145, 0.0058)
ax.set_xticks(np.arange(-0.012, 0.0061, 0.002))
ax.xaxis.set_major_formatter(FMT(3))
ax.set_xlabel("Variation de Gini vs la référence, comparée pli par pli\n(barre = erreur-type ; entre parenthèses : plis gagnés sur 5)")
ax.set_title("Ce qui a marché, ce qui n'a pas marché")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=VERT, label="gardé"), Patch(color=GRIS, label="neutre"), Patch(color=ROUGE, label="rejeté")],
          loc="lower left", frameon=False)
ax.grid(axis="y", alpha=0)
sauver("11_ce_qui_a_marche.png")

# ---------------------------------------------------------------- 6. apport des membres
liste = [("E30", "LightGBM\n(Jahrer)"), ("E07", "XGBoost"), ("E08", "CatBoost"), ("E32c", "Réseau + DAE"), ("E33f", "Mélange E33f\n(4 membres)")]
v = [cv(e)[0] for e, _ in liste]; sd = [cv(e)[1] for e, _ in liste]
fig, ax = plt.subplots(figsize=(11, 6.3))
ax.bar(range(5), v, yerr=sd, capsize=6, color=[ORANGE, ORANGE, ORANGE, "#9b59b6", VERT], ecolor="#333", width=0.62)
for i, a in enumerate(v):
    ax.text(i, 0.2335, fr(a, 4), ha="center", fontsize=12, fontweight="bold", color="white")
ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n in liste])
ax.set_ylim(0.23, 0.30); ax.yaxis.set_major_formatter(FMT(2)); ax.set_ylabel("Gini en validation croisée (± écart-type entre plis)")
ax.axhline(max(v[:4]), color=GRIS, ls="--", lw=1)
ax.set_title("Le mélange fait mieux que chacun de ses membres")
d1 = cmp_[(cmp_.exp == "E33f") & (cmp_.ref == "E09")].iloc[0]
ax.text(0.01, 0.97, f"Le réseau est le plus faible seul ({fr(v[3], 3)}), mais il apporte de la diversité :\n"
        f"mélange à 4 vs mélange à 3 boostings : +{fr(d1.d_moy, 4)}, 5 plis sur 5 gagnés.", transform=ax.transAxes, va="top", fontsize=11)
sauver("11_apport_membres.png")
RES["membres"] = {e: round(cv(e)[0], 5) for e, _ in liste}
json.dump(RES, open(RAP / "15_resultats.json", "w"), indent=1, default=lambda o: None)
print(json.dumps(RES, indent=1, default=str))
