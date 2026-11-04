# 15 - Analyste : figures et soutenance pour 2_306 (E33f)

Date : 2026-11-04 ~14h35 Paris. Script : `usine/src/15_figures_2_306.py` (pandas, numpy, matplotlib, CPU, ~10 s ; n'écrit que figures et rapports 15_*). Commande : `python -u usine/src/15_figures_2_306.py` (cwd = Hackathon).

## Chiffres clés (OOF E33f, 446 409 assurés)
- Les 10 % les plus risqués = 22,2 % des sinistres ; **les 20 % = 37,2 %** ; les 50 % = 68,4 %.
- Taux de sinistre : décile 1 = 1,47 %, décile 10 = 8,10 %, **ratio 5,5** (moyenne 3,64 %). Table : `rapports/15_deciles_E33f.csv`.
- Corrélations de rang des OOF : E30-E07 0,963 ; E07-E08 0,962 ; E30-E08 0,941 (arbres : 0,94 à 0,96) ; réseau E32c vs arbres : 0,912 (E30), 0,913 (E07), 0,917 (E08).
- Gini CV seuls : E30 0,2798 ; E07 0,2807 ; E08 0,2815 ; E32c 0,2752 ; mélange E33f 0,2853 ± 0,0074. Mélange 4 vs 3 boostings (E09) : +0,0019, 5/5 plis.
- Gini CV des soumissions : 2_302 0,2787 ; 2_303 0,2834 ; 2_305 0,2843 ; 2_306 0,2853 ; 2_307 0,2856. Serveur : 2_306 0,30079 ; 2_307 0,3003.

## Écarts avec la consigne (à relayer)
- Les corrélations mesurées sont **0,91-0,92 (réseau) contre 0,94-0,96 (arbres)**, pas 0,87-0,91 contre 0,96-0,99 : la figure et les documents utilisent les valeurs calculées.
- L'AUC du modèle final est environ 0,64 (Gini 0,285), pas 0,65.
- Le décile 10/1 est 5,5 (E09 : 5,4 dans l'ancien plan).
- Figure `11_ce_qui_a_marche` : pour E27 (3 graines) j'ai utilisé E28 vs E02m (+0,0005, 3/5, neutre) ; pour les autres, la ligne de `comparaisons.csv` avec la référence indiquée sur chaque barre (références différentes : E01, E02, E09, E13, E17, E29, E33f, E52_g0). Barre d'erreur = écart-type entre plis / racine de 5. Données : `rapports/15_ce_qui_a_marche.csv`.
- E52 est un MLP sans DAE (repli) ; E41 = mélange avec le DAE repris (E40c).
- Axe de la figure 6 tronqué à 0,23 (barres) : préciser à l'oral que les écarts sont petits (< 0,01).

## Figures (`usine/figures/11_*.png`, relues une par une)
`11_progression_soumissions.png`, `11_schema_modele_final.png`, `11_correlations_membres.png`, `11_gains_deciles_E33f.png`, `11_ce_qui_a_marche.png`, `11_apport_membres.png`.

## Soutenance
- `usine/soutenance/plan.md` : 12 slides, 14 min, répartition imposée en 5 membres, notes orales, figures, récit des échecs, préparation jusqu'à 16h30 avec 2 questions par membre. Ancienne version sauvegardée (`plan_ancien_E09.md.bak`).
- `usine/soutenance/questions_reponses.md` : 22 questions attribuées, dont 2_306 vs 2_307 (Q9), serveur > CV (Q10), Kaggle (Q5), autoencodeur et test (Q8).

## Points de vigilance pour le jury
- SHAP et profils (figures 05_*) viennent du LightGBM de la baseline (E01), pas du mélange final : à dire explicitement (Q18).
- L'autoencodeur est entraîné sur train + test sans cible (transductif) : annoncé dans les limites, slide 12.
- Le Membre 1 doit confirmer 446 409 / 148 803 (repris de l'ancien plan).
- La mention « Cinq membres, qui a fait quoi » est à compléter avec les prénoms.
