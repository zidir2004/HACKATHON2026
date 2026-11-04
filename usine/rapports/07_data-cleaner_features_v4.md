# 07 - Data cleaner : feature engineering, version v4

## Mission
v4 = v1 (37 variables, sans ps_calc_*) + 12 variables candidates (interactions, comptages, recombinaisons). Aucune utilisation de la cible.

## Résultat
- Code : `usine/src/02b_features.py` (une fonction par famille, rejouable ; job `12_features_v4`, log OK).
- Données : `usine/data/processed/v4/{train,test}.parquet` + `schema.json` (49 variables, clé `new_features` = famille de chaque nouvelle variable). Gini univarié : `usine/data/processed/v4_gini_univarie.csv`.
- Contrôles automatiques passés : mêmes lignes et ordre que v1 (446 409 / 148 803), ids et target identiques, colonnes v1 inchangées, mêmes colonnes train/test, 0 NaN, 0 inf, -1 cohérents (voir plus bas), entiers reconstruits à 1e-4 près de la variable d'origine.
- Manquants : aucun nouveau NaN ; les -1 de v1 sont conservés ; taux de -1 des variables d'origine inchangés.
- Différentiel vs v1 : +12 colonnes, 0 retirée.

## Point d'attention : statistiques train+test
Les comptages (F1, F5) sont calculés sur train+test concaténés. C'est non supervisé (aucune cible), donc pas de fuite de la cible, mais la distribution du test est utilisée. Légitime en compétition ; à signaler si on présente la méthode.

## Variables, formules, justification, Gini univarié
Gini = 2*AUC-1 de la variable brute comme score, sur le train, folds `folds.csv`, moyenne des 5 folds (écart-type entre folds ~0,005 à 0,015). Signe libre, on lit la valeur absolue.

| Fam. | Variable | Formule | Justification | Gini univarié (|moy|) | Réf. variable source |
|---|---|---|---|---|---|
| F1 | ps_car_11_cat_count | fréquence de la modalité (train+test) | fréquence = proxy de popularité du modèle de voiture, que l'arbre ne voit pas | 0,021 | ps_car_11_cat 0,026 |
| F1 | ps_car_01_cat_count | idem | idem | 0,011 | 0,057 |
| F1 | ps_car_06_cat_count | idem | idem | 0,072 | 0,033 |
| F2 | reg01_x_reg02 | ps_reg_01 * ps_reg_02 | interaction des deux variables régionales | 0,094 | 0,066 / 0,098 |
| F2 | car13_x_reg03 | ps_car_13 * ps_reg_03 ; -1 si ps_reg_03 = -1 | combine les 2 plus fortes variables continues | 0,145 | 0,150 / 0,110 |
| F2 | car13_div_car12 | ps_car_13 / ps_car_12 ; -1 si ps_car_12 <= 0 (1 ligne) | ratio, difficile à approcher par coupures successives | 0,117 | 0,150 / 0,111 |
| F2 | ind03_plus_ind15 | ps_ind_03 + ps_ind_15 | les deux sont forts en SHAP | 0,032 | 0,026 / 0,066 |
| F2 | ind03_x_ind15 | ps_ind_03 * ps_ind_15 | idem | 0,024 | idem |
| F3 | ps_car_15_int | round(ps_car_15^2) | entier d'origine (racine) ; transformation monotone, effet nul attendu pour les arbres | 0,088 | = ps_car_15 |
| F3 | ps_reg_03_int | round((ps_reg_03*40)^2), -1 conservé ; erreur de reconstruction max 2e-7 | idem ; utile pour l'oral | 0,110 | = ps_reg_03 |
| F4 | ps_ind_bin_sum | somme des 11 ps_ind_*_bin | compteur de binaires à 1 | 0,009 | - |
| F5 | ind05_x_car04_count | comptage de la paire (ps_ind_05_cat, ps_car_04_cat), 62 paires | recombinaison de 2 catégorielles | 0,124 | 0,043 / 0,075 |

## Lecture pour prioriser
- Plus forts en univarié : car13_x_reg03 (0,145, mais ~ égal à ps_car_13 seule : information probablement redondante), ind05_x_car04_count (0,124, bien au-dessus de ses deux sources 0,043 et 0,075 : le plus prometteur, car il vient d'une recombinaison), car13_div_car12 (0,117 ; ps_car_12 seule 0,111).
- ps_car_06_cat_count (0,072 contre 0,033 pour la source) : gain univarié, à tester.
- F3 identique à la variable d'origine (même rang), F4 quasi nul : faible priorité. F2 ind03_* : faible Gini mais ce sont les interactions de variables fortes en SHAP, l'univarié ne dit pas grand-chose de leur apport conditionnel.
- Un Gini univarié ne prédit pas le gain dans le modèle (les arbres captent déjà les variables seules) ; seul le test en CV apparié tranche. Un gain < 0,005 reste dans le bruit.

## Pas appliqué, à envisager si v4 est utile
Croisements supplémentaires (ps_car_04_cat x ps_car_01_cat...), encodage cible du croisement (dans les folds), comptages par paire de catégorielles, nombre de -1 (v2 neutre, déjà écarté).
