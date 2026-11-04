"""10_analyse_approfondie.py — figures de distribution, de progression et de performance finale (E09).

Usage (depuis le dossier Hackathon) :  python -u usine/src/10_analyse_approfondie.py
Sorties : usine/figures/10_*.png, usine/rapports/10_*.csv, usine/rapports/10_resultats.json
Aucune dépendance à scipy/shap : pandas + numpy + matplotlib seulement. N'utilise pas le GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[2]
USINE = RACINE / "usine"
FIG, RAP, EXP = USINE / "figures", USINE / "rapports", USINE / "experiences"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "axes.titleweight": "bold"})
BLEU, ORANGE, GRIS, ROUGE, VERT = "#1f5fa8", "#e07b00", "#8a8a8a", "#c0392b", "#2e8b57"
COUL_GROUPE = {"ind": BLEU, "reg": VERT, "car": ORANGE, "calc": GRIS}
RES: dict = {}


def sauver(nom: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / nom, bbox_inches="tight")
    plt.close()
    print("  figure", nom, flush=True)


def wilson(k, n, z=1.96):
    """Intervalle de confiance de Wilson à 95 % pour une proportion k/n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


train = pd.read_csv(RACINE / "train.csv")
y = train["target"].to_numpy()
TAUX = y.mean()
RES["n"], RES["taux"] = int(len(train)), float(TAUX)

# ---------------------------------------------------------------------------
# 1. Variables continues : distribution + taux de sinistre par tranche
# ---------------------------------------------------------------------------
CONT = ["ps_car_13", "ps_reg_03", "ps_car_12", "ps_car_14", "ps_ind_03", "ps_ind_15", "ps_reg_01", "ps_reg_02"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
lignes = []
for i, (ax, v) in enumerate(zip(axes.ravel(), CONT)):
    s = train[v]
    manq = s == -1
    ok = ~manq
    x, yy = s[ok].to_numpy(), y[ok]
    if len(np.unique(x)) <= 25:                      # variable entière / ordinale : une barre par valeur
        vals = np.unique(x)
        idx = np.searchsorted(vals, x)
        centres = vals
        larg = 0.8 * (np.min(np.diff(vals)) if len(vals) > 1 else 1)
    else:                                           # variable fine : 20 tranches de même largeur (0,5-99,5 %)
        lo, hi = np.percentile(x, [0.5, 99.5])
        bords = np.linspace(lo, hi, 21)
        idx = np.clip(np.searchsorted(bords, x, side="right") - 1, 0, 19)
        centres = (bords[:-1] + bords[1:]) / 2
        larg = (bords[1] - bords[0]) * 0.9
    n_b = np.bincount(idx, minlength=len(centres))
    k_b = np.bincount(idx, weights=yy, minlength=len(centres))
    part = 100 * n_b / len(x)
    taux = np.where(n_b >= 300, 100 * k_b / np.maximum(n_b, 1), np.nan)
    ax.bar(centres, part, width=larg, color=BLEU, alpha=0.35)
    ax.set_xlabel(v)
    if i % 4 == 0:
        ax.set_ylabel("Part des assurés (%)")
    ax2 = ax.twinx()
    ax2.plot(centres, taux, color=ORANGE, marker="o", ms=4, lw=2)
    ax2.axhline(100 * TAUX, color=GRIS, ls="--", lw=1)
    ax2.set_ylim(0, np.nanmax(taux) * 1.25)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    if i % 4 == 3:
        ax2.set_ylabel("Taux de sinistre (%)", color=ORANGE)
    sous = ""
    if manq.sum() > 100:
        sous = f"\nmanquant (-1) : {100*manq.mean():.0f} % des assurés, taux {100*y[manq].mean():.2f} %".replace(".", ",")
    ax.set_title(v + sous, fontsize=11)
    for c, nb, kb in zip(centres, n_b, k_b):
        lignes.append({"variable": v, "tranche_centre": float(c), "n": int(nb), "sinistres": int(kb),
                       "taux": float(kb / nb) if nb else np.nan})
    if manq.sum():
        lignes.append({"variable": v, "tranche_centre": -1, "n": int(manq.sum()), "sinistres": int(y[manq].sum()),
                       "taux": float(y[manq].mean())})
fig.suptitle("Variables continues : distribution (barres bleues) et taux de sinistre par tranche (courbe orange ; "
             "tirets gris = taux moyen 3,64 %)", fontsize=13, fontweight="bold")
sauver("10_continues.png")
pd.DataFrame(lignes).to_csv(RAP / "10_continues_tranches.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Catégorielles / binaires : taux par modalité avec IC 95 % et effectifs
# ---------------------------------------------------------------------------
CAT = ["ps_ind_05_cat", "ps_car_07_cat", "ps_car_01_cat", "ps_ind_04_cat",
       "ps_car_04_cat", "ps_car_09_cat", "ps_ind_17_bin", "ps_ind_06_bin"]
fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
lignes = []
for ax, v in zip(axes.ravel(), CAT):
    g = train.groupby(v)["target"].agg(["size", "sum"]).reset_index()
    lo, hi = wilson(g["sum"], g["size"])
    p = 100 * g["sum"] / g["size"]
    xs = np.arange(len(g))
    manq = g[v] == -1
    ax.errorbar(xs, p, yerr=[100 * (g["sum"] / g["size"]) - 100 * lo, 100 * hi - 100 * (g["sum"] / g["size"])],
                fmt="none", ecolor=GRIS, elinewidth=2, capsize=4, zorder=1)
    ax.scatter(xs[~manq], p[~manq], color=BLEU, s=60, zorder=3)
    ax.scatter(xs[manq], p[manq], color=ROUGE, s=70, marker="D", zorder=3)
    ax.axhline(100 * TAUX, color=GRIS, ls="--", lw=1)
    ymax = 100 * hi[~manq.to_numpy()].max() * 1.12
    if manq.any() and 100 * hi[manq.to_numpy()].min() > 2.2 * ymax:      # modalité manquante rare : hors échelle
        ax.set_ylim(0, ymax)
        ax.text(0.5, 0.6, f"manquant : {p[manq].iloc[0]:.0f} % (n = {int(g['size'][manq].iloc[0])}),\nhors échelle",
                transform=ax.transAxes, ha="center", va="center", color=ROUGE, fontsize=10)
    etiq = [("manq." if m == -1 else str(m)) + "\n" + (f"{n/1000:.0f}k" if n >= 10000 else f"{n/1000:.1f}k")
            for m, n in zip(g[v], g["size"])]
    ax.set_xticks(xs)
    ax.set_xticklabels(etiq, fontsize=7.5 if len(g) > 8 else 10)
    ax.set_xlim(-0.6, len(g) - 0.4)
    ax.set_xlabel(v + "   (modalité / effectif en milliers)", fontsize=10)
    ax.set_ylabel("Taux de sinistre (%)")
    ax.set_title(v)
    for m, n, k in zip(g[v], g["size"], g["sum"]):
        lignes.append({"variable": v, "modalite": int(m), "n": int(n), "sinistres": int(k), "taux": k / n})
fig.suptitle("Taux de sinistre par modalité, avec intervalle de confiance à 95 % (losange rouge = valeur manquante ; "
             "tirets = taux moyen 3,64 %)", fontsize=13, fontweight="bold")
sauver("10_categorielles.png")
pd.DataFrame(lignes).to_csv(RAP / "10_categorielles_modalites.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Panorama du signal par groupe (ind / reg / car / calc)
# ---------------------------------------------------------------------------
av = pd.read_csv(RAP / "audit_variables.csv")
shap = pd.read_csv(RAP / "05_shap_importance.csv").rename(columns={"Unnamed: 0": "var"})
shap["groupe"] = shap["var"].str.split("_").str[1]
part_shap = (shap.groupby("groupe")["shap_abs_moyen"].sum() / shap["shap_abs_moyen"].sum() * 100).to_dict()
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
tailles = {}
for ax, g in zip(axes.ravel(), ["ind", "reg", "car", "calc"]):
    d = av[av["groupe"] == g].sort_values("signal")
    ax.barh(d["var"], d["signal"], color=COUL_GROUPE[g])
    ax.set_title(f"{g} : {len(d)} variables, {part_shap[g]:.0f} % de l'importance SHAP du modèle", fontsize=11.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0)
    tailles[g] = {"n_variables": int(len(d)), "signal_median": float(d["signal"].median()),
                  "signal_max": float(d["signal"].max()), "part_shap_pct": float(part_shap[g])}
for ax in axes[1]:
    ax.set_xlabel("Signal univarié : Gini en valeur absolue")
fig.suptitle("Signal par groupe de variables : ind, reg et car portent l'information, les 20 ps_calc sont du bruit",
             fontsize=13, fontweight="bold")
sauver("10_signal_groupes.png")
RES["groupes"] = tailles

# ---------------------------------------------------------------------------
# 4. Modèle final E09 : gains, déciles, distribution des scores ; comparaison E00 / E01 / E09
# ---------------------------------------------------------------------------
def charger_oof(e):
    d = pd.read_csv(EXP / "oof" / f"{e}_oof.csv").sort_values("id", ignore_index=True)
    return d


def gains(d):
    o = np.argsort(-d["oof"].to_numpy(), kind="stable")
    cum = np.cumsum(d["target"].to_numpy()[o]) / d["target"].sum()
    return (np.arange(1, len(d) + 1) / len(d)), cum


def gini_oof(d):
    r = d["oof"].rank().to_numpy()
    n1 = d["target"].sum()
    n0 = len(d) - n1
    auc = (r[d["target"].to_numpy() == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return 2 * auc - 1, auc


oof = {e: charger_oof(e) for e in ["E00", "E01", "E09"]}
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], color=GRIS, ls="--", label="Hasard")
for e, c, lab in [("E00", GRIS, "E00 régression logistique"), ("E01", BLEU, "E01 LightGBM seul"),
                  ("E09", ORANGE, "E09 mélange final")]:
    fx, fy = gains(oof[e])
    ax.plot(fx, fy, color=c, lw=2.5 if e == "E09" else 1.8, label=f"{lab} (Gini {gini_oof(oof[e])[0]:.3f})".replace(".", ","))
res_g = {}
for e in ["E00", "E01", "E09"]:
    fx, fy = gains(oof[e])
    res_g[e] = {f"top{int(p*100)}": float(fy[int(p * len(fx)) - 1]) for p in (0.1, 0.2, 0.3, 0.5)}
    res_g[e]["gini_oof"], res_g[e]["auc_oof"] = [float(t) for t in gini_oof(oof[e])]
fx, fy = gains(oof["E09"])
t20 = res_g["E09"]["top20"]
ax.plot([0.2, 0.2, 0], [0, t20, t20], color=ORANGE, ls=":", lw=1.5)
ax.annotate(f"20 % des assurés les plus risqués\n= {100*t20:.0f} % des sinistres", (0.2, t20), (0.33, t20 - 0.17),
            arrowprops=dict(arrowstyle="->", color=ORANGE), fontsize=11.5, color=ORANGE)
ax.set_xlabel("Part des assurés, classés du plus au moins risqué selon le modèle")
ax.set_ylabel("Part des sinistres capturée")
ax.set_title("Courbe de gains (prédictions hors échantillon, 446 409 assurés)")
ax.legend(loc="lower right", fontsize=10.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
sauver("10_gains_E09.png")
RES["gains"] = res_g

d9 = oof["E09"].copy()
d9["decile"] = pd.qcut(d9["oof"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec = d9.groupby("decile")["target"].agg(["size", "sum", "mean"]).reset_index()
dec.columns = ["decile", "n", "sinistres", "taux"]
lo, hi = wilson(dec["sinistres"], dec["n"])
dec.to_csv(RAP / "10_deciles_E09.csv", index=False)
d1 = oof["E01"].copy()
d1["decile"] = pd.qcut(d1["oof"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
dec1 = d1.groupby("decile")["target"].mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(dec["decile"], 100 * dec["taux"], color=ORANGE, yerr=[100 * (dec["taux"] - lo), 100 * (hi - dec["taux"])],
       capsize=3, ecolor="#444")
ax.axhline(100 * TAUX, color=GRIS, ls="--")
ax.text(0.55, 100 * TAUX + 0.15, "taux moyen 3,64 %", color="#555", fontsize=10.5)
for x_, t in zip(dec["decile"], dec["taux"]):
    ax.text(x_, 100 * t + 0.45, f"{100*t:.1f}".replace(".", ","), ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(range(1, 11))
ax.set_xlabel("Décile du score (1 = 10 % les moins risqués, 10 = 10 % les plus risqués)")
ax.set_ylabel("Taux de sinistre observé (%)")
ratio = dec["taux"].iloc[-1] / dec["taux"].iloc[0]
ax.set_title(f"Le taux de sinistre est multiplié par {ratio:.1f}".replace(".", ",") + f" entre le décile 1 et le décile 10 (modèle final E09)")
sauver("10_deciles_E09.png")
RES["deciles"] = {"taux_d1": float(dec["taux"].iloc[0]), "taux_d10": float(dec["taux"].iloc[-1]), "ratio": float(ratio),
                  "taux_d10_E01": float(dec1.iloc[-1]), "taux_d1_E01": float(dec1.iloc[0]),
                  "part_sinistres_5_deciles_bas": float(dec["sinistres"].iloc[:5].sum() / dec["sinistres"].sum()),
                  "part_sinistres_3_deciles_bas": float(dec["sinistres"].iloc[:3].sum() / dec["sinistres"].sum())}

# distribution du score par classe (le chevauchement = la limite du modèle)
pct = d9["oof"].rank(pct=True) * 100
fig, ax = plt.subplots(figsize=(10, 5.8))
bins = np.linspace(0, 100, 26)
ax.hist(pct[d9["target"] == 0], bins=bins, density=True, alpha=0.55, color=BLEU, label="Sans sinistre (96,4 %)")
ax.hist(pct[d9["target"] == 1], bins=bins, density=True, alpha=0.6, color=ORANGE, label="Avec sinistre (3,6 %)")
ax.axhline(1 / 100, color=GRIS, ls="--", lw=1)
ax.set_xlabel("Rang de l'assuré selon le score du modèle (0 = peu risqué, 100 = très risqué)")
ax.set_ylabel("Densité")
ax.set_title("Les deux populations se chevauchent fortement : le modèle trie, il ne tranche pas (AUC 0,64)")
ax.legend(fontsize=11)
sauver("10_scores_par_classe.png")

# progression des modèles : Gini CV moyen ± écart-type entre folds
reg = pd.read_csv(EXP / "registre.csv").set_index("id")
cles = [("E00", "E00\nrégression\nlogistique", GRIS), ("E01", "E01\nLightGBM,\ndonnées brutes", BLEU),
        ("E02", "E02\nsans les\n20 ps_calc", BLEU), ("E06", "E06\ncatégorielles\nnatives (échec)", ROUGE),
        ("E09", "E09\nmélange de 3\nalgorithmes", ORANGE)]
comp = pd.read_csv(EXP / "comparaisons.csv")
fig, ax = plt.subplots(figsize=(11, 6.2))
for i, (e, lab, c) in enumerate(cles):
    m, s = reg.loc[e, "score_cv_moyen"], reg.loc[e, "score_cv_std"]
    ax.bar(i, m, color=c, yerr=s, capsize=6, ecolor="#333")
    ax.text(i, m + s + 0.004, f"{m:.4f}".replace(".", ","), ha="center", fontweight="bold", fontsize=12)
notes = {"E01": "+0,039 vs E00", "E02": "+0,0022 vs E01\n(4/5 folds,\nnon signif.)", "E06": "−0,0099 vs E02\n(0/5 folds)",
         "E09": "+0,0026 vs E02\n(5/5 folds)"}
for i, (e, _, _) in enumerate(cles):
    if e in notes:
        ax.text(i, 0.03, notes[e], ha="center", color="white", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(cles)))
ax.set_xticklabels([c[1] for c in cles])
ax.set_ylim(0, 0.34)
ax.set_ylabel("Gini en validation croisée (5 folds)")
ax.set_title("Progression : le saut vient des arbres, puis du mélange de modèles")
ax.set_xlabel("Barres d'erreur = écart-type entre folds (≈ 0,008) : les différences se jugent fold par fold", fontsize=10, color="#555")
sauver("10_progression_modeles.png")
RES["progression"] = {e: [float(reg.loc[e, "score_cv_moyen"]), float(reg.loc[e, "score_cv_std"])] for e, _, _ in cles}

# ce qui a marché / pas marché : Δ apparié par fold avec barre d'erreur ±2 erreurs-types
LIB = {"E02": "retrait des 20 ps_calc", "E03": "+ indicateurs de manquants", "E04": "redondances nettoyées",
       "E05": "encodage cible de ps_car_11_cat", "E06": "catégorielles natives LightGBM", "E07": "XGBoost seul",
       "E08": "CatBoost seul", "E09": "mélange LightGBM + XGBoost + CatBoost", "E10": "LightGBM : feature_fraction 0,4",
       "E11": "LightGBM : num_leaves 16", "E12": "LightGBM : min_child_samples 500", "E13": "LightGBM : pas d'apprentissage 0,01",
       "E14": "mélange avec E13", "E15": "CatBoost : pas plus faible", "E16": "XGBoost : pas plus faible",
       "E17a": "LightGBM moyenné sur 3 graines", "E17": "mélange avec 3 graines"}
def libelle(e):
    return LIB.get(e) or str(reg.loc[e, "changement"])[:58]


c = comp[~((comp.exp == "E04") & (comp.ref == "E03")) & ~((comp.exp == "E17") & (comp.ref == "E14"))].copy()
fol = c[[f"d_fold{k}" for k in range(5)]].to_numpy()
c["moy"] = fol.mean(1)
c["se2"] = 2 * fol.std(1, ddof=1) / np.sqrt(5)
c["gagnes"] = (fol > 0).sum(1)
def verdict(r):
    if r.moy > 0 and (r.gagnes == 5 or r.moy > r.se2):
        return "garde"
    if r.moy < 0 and (r.gagnes == 0 or -r.moy > r.se2):
        return "rejete"
    return "neutre"
c["verdict_regle_durcie"] = c.apply(verdict, axis=1)
c = c.sort_values("moy").reset_index(drop=True)
c.to_csv(RAP / "10_ce_qui_a_marche.csv", index=False)
fig, ax = plt.subplots(figsize=(13, max(9, 0.36 * len(c) + 2)))
cm = {"garde": VERT, "rejete": ROUGE, "neutre": GRIS}
for i, r in c.iterrows():
    ax.errorbar(r.moy, i, xerr=r.se2, fmt="o", color=cm[r.verdict_regle_durcie], ms=8, capsize=4, lw=2)
ax.axvline(0, color="#333", lw=1)
ax.set_yticks(range(len(c)))
ax.set_yticklabels([f"{r.exp} {libelle(r.exp)}  [réf. {r.ref}]  ({r.gagnes}/5)" for r in c.itertuples()], fontsize=10)
ax.set_xlabel("Gain de Gini vs la référence : moyenne des 5 folds ± 2 erreurs-types\n(entre parenthèses : nombre de folds gagnés)")
ax.set_title("Ce qui a marché (vert), ce qui a échoué (rouge) et ce qui ne change rien (gris)")
ax.grid(axis="y", alpha=0)
for lab, col in [("Gain réel", VERT), ("Sans effet mesurable", GRIS), ("Dégradation", ROUGE)]:
    ax.plot([], [], "o", color=col, label=lab)
ax.legend(loc="upper left", fontsize=11)
sauver("10_ce_qui_a_marche.png")
RES["verdicts"] = {f"{r.exp} vs {r.ref}": r.verdict_regle_durcie for r in c.itertuples()}

# ---------------------------------------------------------------------------
# 5. Vision métier : segmentation par score (E09)
# ---------------------------------------------------------------------------
seg = pd.cut(d9["oof"].rank(pct=True), [0, 0.5, 0.8, 0.9, 1.0], labels=["50 % les moins risqués", "30 % intermédiaires",
                                                                         "10 % suivants", "10 % les plus risqués"])
sg = d9.groupby(seg, observed=True)["target"].agg(["size", "sum", "mean"])
sg.to_csv(RAP / "10_segments_E09.csv")
RES["segments"] = {k: {"n": int(v["size"]), "taux": float(v["mean"]), "part_sinistres": float(v["sum"] / y.sum())}
                   for k, v in sg.iterrows()}
(RAP / "10_resultats.json").write_text(json.dumps(RES, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps({k: RES[k] for k in ["gains", "deciles", "segments", "groupes"]}, indent=1, ensure_ascii=False))
