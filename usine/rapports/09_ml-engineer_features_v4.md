# 09 - ML engineer : test de la v4 (feature engineering), E19-E28

## Hypothèse
Les 12 variables candidates du data cleaner (comptages, ratios/produits, recombinaisons) apportent
une information que les arbres n'approchent pas seuls (rapport 07, Gini univarié élevé pour F5 et F2).

## Protocole
- Mêmes folds (`usine/experiences/folds.csv`), LightGBM config **E02** (lr 0,02, ~40 s par expérience, CPU n_jobs 16).
- Ablation par famille **sans nouveau parquet** : on lit v4 et on exclut les candidates non testées
  (`03_train.v4_avec([...])`). Référence appariée = E02 (v1). Règle : `garde` si Δ>0 et (5/5 ou Δ > 2·sd(Δ)/√5).
- Nouveauté : vérification de robustesse à la **graine du modèle** (`usine/src/09_verif_graines.py`) avant
  de croire un « garde » obtenu avec une seule graine.
- Code : `usine/src/03_train.py` (configs E19-E22, E25-E27 ; `--familles-v4` -> E24x/E24c ;
  `--membres-graines`), `06_melange.py` (E24), `09_verif_graines.py` (E28). Jobs 18 à 22.

## Résultats (Gini, moyenne ± écart-type sur 5 folds ; Δ apparié par fold)

| Exp | Changement | Gini CV | Réf | Δ moy | Folds gagnés | Verdict |
|---|---|---|---|---|---|---|
| E02 | réf. LightGBM v1 | 0,28088 ± 0,0080 | - | - | - | référence |
| E19 | v4 complète (+12) | 0,27805 ± 0,0073 | E02 | −0,00283 | 0/5 | **rejeté** |
| E20 | v1 + F5 ind05_x_car04_count | 0,28072 ± 0,0074 | E02 | −0,00016 | 3/5 | neutre |
| E21 | v1 + F2 car13_div_car12, car13_x_reg03 | 0,27958 ± 0,0068 | E02 | −0,00129 | 1/5 | **rejeté** |
| E22 | v1 + F1 (3 comptages) | 0,28113 ± 0,0078 | E02 | +0,00025 | 2/5 | neutre |
| E25 | v1 + F2b (reg01×reg02, ind03±ind15) | 0,28088 ± 0,0080 | E02 | +0,00000 | 3/5 | neutre |
| E26 | v1 + F3 (*_int) | 0,28050 ± 0,0078 | E02 | −0,00038 | 2/5 | neutre |
| E27 | v1 + F4 ps_ind_bin_sum | 0,28211 ± 0,0072 | E02 | +0,00123 | 5/5 | garde (graine 42 seule) |
| E28 | E27 vs E02, **moyenne 3 graines** (42/7/2026) | 0,28241 ± 0,0078 | E02m | +0,00050 | 3/5 | **neutre** |
| E24x | XGBoost GPU (E07) sur v1 + F4 | 0,28093 ± 0,0081 | E07 | +0,00020 | 3/5 | neutre |
| E24c | CatBoost GPU (E08) sur v1 + F4 | 0,28040 ± 0,0067 | E08 | −0,00111 | 0/5 | **rejeté** |
| E24 | rangs égaux E27+E24x+E24c | 0,28349 ± 0,0074 | E09 | +0,00007 | 3/5 | neutre |

Pas de seuil (Gini = 2·AUC−1). Temps : ~40 s par LightGBM, 18 s XGBoost GPU, 425 s CatBoost GPU ; ~20 min au total.

## Interprétation
- **E19 (v4 complète, −0,0028, 0/5)** : ajouter 12 variables en bloc dégrade. Les variables fortement corrélées à
  ps_car_13/ps_reg_03 (F2) et les copies monotones (F3) diluent le tirage de colonnes (feature_fraction 0,6) et
  offrent des coupures redondantes.
- **E20 (F5)** : le Gini univarié élevé (0,124) ne se traduit pas en gain : l'arbre reconstruit déjà le
  croisement ps_ind_05_cat × ps_car_04_cat par deux coupures successives. Neutre.
- **E21 (F2 car13)** : rejeté (−0,0013, 1/5). car13_x_reg03 ≈ ps_car_13 (même Gini univarié) : redondance pure ;
  le ratio ne dit rien de plus que les deux variables.
- **E22 (F1 comptages)** : +0,00025, 2/5 : bruit. ps_car_11_cat est déjà lue comme nombre (cf. E05/E06).
- **E25/E26** : nuls, comme prévu par le data cleaner (F3 est monotone, donc invisible pour un arbre).
- **E27 (F4) puis E28 : faux positif.** E27 passe la règle avec la graine 42 (5/5, +0,0012), mais avec les
  graines 7 et 2026 le Δ vaut −0,00044 (2/5) et +0,00048 (2/5). En moyennant 3 graines de chaque côté (E28) :
  +0,0005, 3/5, sous le seuil 2se (0,0007) -> **neutre**. Le bruit de graine sur un Δ apparié (sd ≈ 0,002 par fold)
  est du même ordre que les effets cherchés : avec 7 familles testées, un 5/5 par hasard n'est pas surprenant
  (p ≈ 1/32 par test).
- **E24** : appliquée par la mission (E27 avait passé la règle) : XGBoost neutre, CatBoost rejeté 0/5,
  mélange +0,00007 vs E09 sur 3/5 -> **neutre, pas de soumission 2_305** (condition « plus de 3/5 » non remplie).

## Verdict
**Aucune variable v4 n'est retenue.** Les arbres captent déjà ces interactions ; v1 reste la version de données.
Référence inchangée : E09 (0,28343) / candidat E17 (0,28371, 2_304).

## Leçon de méthode (à reprendre à l'oral)
Pour un changement de **features**, une seule graine de modèle ne suffit pas : faire la comparaison appariée sur
la moyenne de 3 graines (`09_verif_graines.py`, ~2 min en LightGBM) avant de déclarer « garde ».

## Prochaine idée recommandée
Le gain ne viendra plus des variables. Restent : (1) poids du mélange ou stacking logistique sur les OOF
(gain attendu ≤ 0,0003), (2) un modèle plus différent pour le mélange (ex. réseau / régression logistique avec
encodage cible) : la corrélation des rangs entre les 3 boostings est 0,96-0,99, d'où le plafond du mélange.
Sinon : figer E17/E09 et consacrer le temps restant à la soutenance.
