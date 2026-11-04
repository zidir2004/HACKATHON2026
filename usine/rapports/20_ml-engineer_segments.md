# 20 — ML engineer — Sélection de modèle par segment (E95–E99)

**Hypothèse (idée du professeur)** : dans certains segments d'assurés un modèle (XGBoost, réseau…) est meilleur ; utiliser le meilleur par segment battrait le mélange E33f.
**Changement** : aucun réentraînement ; candidats E30, E07, E08, E32c, E33f convertis en rangs percentiles globaux ; segmentations fixées a priori ; choix du meilleur par segment appris sur 4 folds, appliqué au 5e (emboîté). Référence E33f (Gini CV 0,28531), comparaison appariée par fold. Script `usine/src/20_selection_segments.py`, log `usine/logs/20_selection_E95_v3.log`.

| Exp | Segmentation | Gini CV | Δ vs E33f | Folds gagnés | Verdict |
|---|---|---|---|---|---|
| E95 | nb manquants (0-1 : 47 % / ≥2 : 53 %, « ≥3 » fusionné car < 15 %) | 0,28531 | +0,00000 | 0/5 (identique) | neutre |
| E96 | tertiles ps_car_13 | 0,28531 | +0,00000 | 0/5 (identique) | neutre |
| E97 | ps_ind_05_cat (-1/0 : 90 % / autre : 10 %, < 15 % inévitable) | 0,28531 | +0,00000 | 0/5 (identique) | neutre |
| E98 | doux top-2 sur E95 | 0,28428 | −0,00102 | 0/5 | rejeté |
| E99 | soumission 2_311 (E95 appliqué au test) | — | — | — | info |

**Résultat** : dans **tous les segments et tous les folds**, le meilleur candidat est le **mélange E33f** (choix 100 % stable). La sélection retombe donc exactement sur E33f.
**Diagnostic (sans le mélange)** : si l'on choisit seulement parmi les 4 modèles, CatBoost gagne souvent (peu de manquants, ps_car_13 extrêmes), XGBoost le tertile central de ps_car_13, le réseau le segment ps_ind_05_cat ≠ 0 — mais le Gini tombe à 0,2809–0,2813 (Δ −0,0040 à −0,0044 vs E33f). Les spécialistes par segment ne valent pas la moyenne : chaque modèle n'est « meilleur » que d'un cheveu, et la moyenne réduit la variance partout.

**Soumission** `usine/submissions/2_311.csv` : contrôlée (148 803 lignes, sans en-tête, ordre test.csv, LF, 0 NaN, [0,0075 ; 0,3946], moyenne 0,0367), **Spearman vs 2_306 = 1,00000** : même classement que 2_306, donc même note serveur attendue (0,30079). La déposer ne coûte qu'une soumission et n'apprend rien de plus.
**Figure oral** : `usine/figures/12_selection_segments.png`. **Temps** : ~1 min CPU.
**Message pour l'oral** : « On a testé l'idée du professeur proprement (choix emboîté) : dans chaque segment, c'est le mélange qui gagne ; la sélection confirme le mélange plutôt que de le remplacer. »
**Prochaine idée** : aucune sur ce thème ; garder 2_306 / 2_307.
