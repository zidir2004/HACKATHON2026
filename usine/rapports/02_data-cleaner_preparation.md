# 02 - Data cleaner : préparation des données (v1, v2, v3)

## Mission reçue
Produire des versions préparées, reproductibles, **une transformation par version**, selon les décisions du point de contrôle n°2 : retirer les 20 ps_calc_* (a), indicateurs de manquants (b), catégorielles bien identifiées (c), nettoyage des redondances (d). Refusé : imputer les -1, écrêter, supprimer des lignes, rééquilibrer.

## Ce qui a été fait
Script `usine/src/02_prepare.py` (une fonction par transformation, rejouable), qui écrit `usine/data/processed/<vX>/{train,test}.parquet` et `schema.json`.

| Version | Transformation (cumulative) | Colonnes (hors id/target) | Dont cat / bin / num |
|---|---|---|---|
| v1 | brut moins les 20 ps_calc_* | 37 | 14 / 11 / 12 |
| v2 | v1 + `nb_manquants` + 5 indicateurs `*_manq` (ps_car_03_cat, ps_car_05_cat, ps_reg_03, ps_car_07_cat, ps_ind_05_cat) | 43 (+6) | 14 / 16 / 13 |
| v3 | v2 - `ps_ind_14` - ps_ind_06..09_bin - ps_ind_16..18_bin + `ps_ind_06_09_cat` + `ps_ind_16_18_cat` | 37 (-6) | 16 / 9 / 12 |

Les parquet gardent l'id (et target pour le train), mêmes lignes et même ordre que `train_raw/test_raw`. Les -1 d'origine sont intacts.

## Résultats clés
- Lignes : train 446 409, test 148 803, inchangées. Aucun NaN introduit.
- Taux de manquants (-1) en train : ps_car_03_cat 69,1 %, ps_car_05_cat 44,8 %, ps_reg_03 18,1 %, ps_car_07_cat 1,9 %, ps_ind_05_cat 1,0 %. Les manquants sont identiques avant/après (rien n'est modifié, seulement ajouté).
- Taux de sinistres quand manquant vs non : car_03 3,28 % / 4,46 % ; car_05 3,18 % / 4,02 % ; reg_03 2,87 % / 3,82 % ; car_07 7,87 % / 3,56 % ; ind_05 8,14 % / 3,60 % (cohérent avec l'audit).
- `nb_manquants` : 0 -> 93 530 lignes, 1 -> 117 219, 2 -> 191 334, 3 -> 42 099, 4 -> 2 164, 6+ -> 63.
- v3 : `ps_ind_14` = somme exacte de ps_ind_10..13_bin (assertion OK, 0 écart). Bijection vérifiée par assertion (une combinaison de binaires <-> un seul code, et inversement).
  - `ps_ind_06_09_cat` : 0 = 175 837 ; 1 = 114 722 ; 2 = 73 051 ; 3 = 82 799 (code = rang de la colonne à 1 parmi 06..09).
  - `ps_ind_16_18_cat` : 0 = 294 976 ; 1 = 54 032 ; 2 = 68 467 ; 3 = 28 934 (3 = aucune des trois).
- Contrôles automatiques (dans le script, tous passés pour v1, v2, v3) : mêmes ids et même ordre, target inchangée, mêmes colonnes train/test hors target, pas de NaN, colonnes conservées identiques à l'origine.

## Décisions et justification
- **Pas d'encodage cible dans les parquet** : ce serait une fuite hors folds. Les `_cat` sont listées dans `schema.json` (`cat_cols`) ; le traitement (natif LightGBM, encodage cible dans les folds) est laissé au ML engineer.
- `nb_manquants` est calculé sur les 37 variables hors calc, y compris les colonnes retirées en v3 (il reste donc identique entre v2 et v3, pour que v3 ne mesure que les redondances).
- Les -1 des `_cat` restent -1 ; `schema.json` indique de les recoder en catégorie à part au moment de l'entraînement (champ `missing_note`), avec la liste `cat_cols_with_missing`.
- Les 5 indicateurs sont des binaires 0/1 (rangés dans `binary_cols`).

## Points d'attention
- v1 vs v0 : l'audit mesurait +0,0017 de Gini en retirant les calc, dans le bruit (±0,0075 par fold) : juger sur la différence appariée par fold.
- Les indicateurs sont redondants avec -1 pour des arbres : gain attendu faible, à mesurer. `nb_manquants` est le plus susceptible d'apporter (Gini -0,072 d'après l'audit).
- v3 : neutre attendu pour LightGBM ; utile surtout pour un modèle linéaire ou CatBoost. `ps_ind_06_09_cat` et `ps_ind_16_18_cat` sont à traiter comme catégorielles (ordre sans sens).
- Les variables `ps_car_11_cat` (104 modalités) et les autres `_cat` : encodage cible uniquement dans les folds.

## Recommandations
1. Lancer la même configuration LightGBM sur v0, v1, v2, v3 avec les folds figés, et comparer par différence appariée par fold.
2. Tester pour les `_cat` : numérique brut vs catégorielle native (en recodant -1) vs encodage cible lissé dans les folds.
3. Pistes non appliquées (non validées) : regroupement des modalités rares, interactions (ps_reg_01 x ps_reg_02, ps_reg_03 x ps_car_13), compteurs par bloc de manquants.

## Fichiers produits
- `usine/src/02_prepare.py` (log : `usine/logs/02_prepare_1.log`)
- `usine/data/processed/v1/`, `v2/`, `v3/` : `train.parquet`, `test.parquet`, `schema.json`
- `usine/data/processed/prep_stats.json` (chiffres du rapport)
- Ce rapport : `usine/rapports/02_data-cleaner_preparation.md`
