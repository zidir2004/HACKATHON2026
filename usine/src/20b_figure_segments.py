"""20b_figure_segments.py — figure pour l'oral de la sélection par segment (relecture lead-tech).

Refait uniquement la figure de 20_selection_segments.py (sans toucher au registre ni à 2_311) :
Gini de chaque modèle dans chaque segment, sur tout l'OOF. Message : dans chaque segment,
le mélange (gris foncé) bat chacun des modèles seuls.
Usage : python -u usine/src/20b_figure_segments.py
"""
import importlib, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
S = importlib.import_module("20_selection_segments")

train, test, y, fold, R, T, probas_test, ids_test = S.charger()
segs = S.segmenter(train, test, S.U.liste_features(train))
titres = {"E95": "Selon le nombre de manquants", "E96": "Selon ps_car_13 (tiers)", "E97": "Selon ps_ind_05_cat"}
couleurs = {"E30": "#4C78A8", "E07": "#F58518", "E08": "#54A24B", "E32c": "#B279A2", "E33f": "#3A3A3A"}
virgule = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, e in zip(axes, ["E95", "E96", "E97"]):
    ctr, _, noms = segs[e]
    ch, gi = S.choisir(y, R, ctr, np.ones(len(y), bool), len(noms), 1)
    w = 0.16
    for j, c in enumerate(S.CANDIDATS):
        xs = np.arange(len(noms)) + (j - 2) * w
        vals = [gi[s][c] for s in range(len(noms))]
        ax.bar(xs, vals, w, color=couleurs[c], label=S.NOMS[c] if e == "E95" else None)
        if c == "E33f":
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(noms)))
    ax.set_xticklabels([f"{n}\n({(ctr == s).mean():.0%} des assurés)".replace(".", ",") for s, n in enumerate(noms)], fontsize=9)
    ax.set_title(titres[e], fontsize=11); ax.grid(axis="y", alpha=.3); ax.set_ylim(0, 0.33)
    ax.yaxis.set_major_formatter(virgule)
axes[0].set_ylabel("Gini dans le segment (prédictions OOF)")
fig.suptitle("Dans chaque segment, le mélange bat chacun des modèles seuls", fontsize=14, fontweight="bold", y=1.0)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5, frameon=False, fontsize=10)
fig.text(0.5, -0.02, "Choisir le meilleur modèle par segment (évaluation emboîtée, 5 folds) retombe toujours sur le mélange : "
         "Gini CV 0,2853, identique à 2_306. Choisir un modèle seul par segment : 0,281 (−0,004).",
         ha="center", fontsize=9.5, color="#555555")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(S.FIG, dpi=140, bbox_inches="tight")
print("figure :", S.FIG)
