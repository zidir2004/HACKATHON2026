# ML engineer — protocole et baseline (2026-11-04 09:20)

## Mission reçue
Fixer le protocole de validation unique du projet, entraîner la baseline E01 (LightGBM sur données brutes v0), un repère naïf E00 (régression logistique), remplir le registre, générer et contrôler la première soumission `2_302.csv`.

## Ce qui a été fait
1. `usine/src/utils.py` : chemins, métrique `gini` (2·AUC−1), chargement par version de données, création/lecture des folds, ajout au registre (delta + verdict automatiques), écriture et contrôle du format de soumission.
2. `usine/src/03_train.py` : une CONFIG par expérience (version de données, features exclues, modèle, paramètres, changement testé) ; boucle CV sur les folds communs ; early stopping sur le fold de validation ; sauvegarde OOF, test et détail par fold ; ligne au registre. Lancement : `python -u usine/src/03_train.py --exp E00 E01`.
3. `usine/src/04_soumission.py` : réaligne les prédictions test d'une expérience sur l'ordre de `test.csv`, écrit la soumission sans en-tête et la contrôle.
4. Folds créés une fois pour toutes : `usine/experiences/folds.csv` (id, fold). Tailles : 89 282 ×4 et 89 281.

## Résultats clés (chiffres)
| Exp | Modèle | Données | Gini CV (moy ± std) | Gini OOF | Durée totale |
|---|---|---|---|---|---|
| E00 | Régression logistique L2 (C=0,1), StandardScaler dans le pipeline | v0 | 0,23923 ± 0,00656 | 0,23917 | 4,5 s |
| E01 | LightGBM | v0 | **0,27870 ± 0,00886** | **0,27856** | 50,9 s |

Détail E01 par fold :

| Fold | Gini | Arbres retenus (early stopping) | Durée |
|---|---|---|---|
| 0 | 0,27657 | 539 | 10,1 s |
| 1 | 0,26470 | 660 | 11,1 s |
| 2 | 0,28818 | 413 | 9,0 s |
| 3 | 0,28836 | 714 | 11,1 s |
| 4 | 0,27568 | 642 | 9,4 s |

E01 vs E00 : +0,0395 de Gini, soit environ 4,5 écarts-types → verdict `garde`. E01 devient la référence.

Soumission `usine/submissions/2_302.csv` : 148 803 lignes, sans en-tête, ids identiques et dans le même ordre que `test.csv`, pas de NaN, valeurs entre 0,0087 et 0,3858, moyenne 0,03641 (taux de positifs du train : 0,03645, cohérent).

## Décisions prises et justification
- **StratifiedKFold à 5 folds, shuffle, random_state=42.** Stratifié parce que la cible est très déséquilibrée (3,64 % de positifs) : chaque fold garde la même proportion de sinistres, sinon le Gini d'un fold varierait juste à cause du nombre de positifs. 5 folds : bon compromis entre la variance de l'estimation (≈ 3 250 positifs par fold de validation) et le temps de calcul. Graine fixe et indices sauvegardés dans `folds.csv` : toutes les expériences sont évaluées sur exactement les mêmes découpages, donc les écarts entre expériences viennent du changement testé et pas du hasard.
- **Pas de CV groupée ni temporelle.** Un id = une ligne, pas de date : rien n'indique d'individus répétés ni d'ordre temporel.
- **Métrique Gini, pas de seuil.** Le Gini ne dépend que de l'ordre des prédictions : il n'y a aucun seuil à optimiser (`seuil = n/a` dans le registre). On rapporte la moyenne ± écart-type des Gini par fold et le Gini OOF global.
- **Règle de verdict** (dans `utils.ajouter_registre`) : gain par rapport au meilleur score > écart-type → `garde` ; perte > écart-type → `rejete` ; sinon `neutre`.
- **Prédiction test** : moyenne des probabilités des 5 modèles de fold, chacun arrêté par early stopping sur son fold de validation. Cela évite de choisir à la main un nombre d'arbres pour un réentraînement sur tout le train, et la moyenne reste dans [0,1].
- **Paramètres E01** (prudents pour un jeu bruité) : learning_rate 0,02 ; num_leaves 24 ; min_child_samples 150 ; feature_fraction 0,6 ; bagging 0,8 (freq 1) ; lambda_l1 1 ; lambda_l2 5 ; 3 000 arbres max ; early stopping 200 ; n_jobs 16 (le PC est partagé avec l'autre agent).
- **v0 strictement brute** : les `_cat` sont traitées comme numériques et −1 reste une valeur ordinaire (LightGBM sait isoler −1 par un découpage). Aucune transformation apprise, donc aucune fuite possible.

## Points d'attention / risques
- L'écart-type entre folds (0,009) est du même ordre que les gains attendus du feature engineering : sur ce jeu, un gain inférieur à ~0,005 doit être pris avec prudence (verdict `neutre`).
- Le nombre d'arbres varie de 413 à 714 selon le fold : le signal est faible et l'optimum est plat.
- Les fichiers OOF sont en CSV (`E<nn>_oof.csv`, colonnes id, target, oof, fold) comme demandé dans la mission, et pas en `.npy`.
- La soumission n'est **pas validée** : le reviewer doit la contrôler avant que l'utilisateur la dépose.

## Recommandations pour la suite
1. E02 : retirer les `ps_calc_*` (piste Kaggle connue : variables a priori de bruit). Un seul changement, à mesurer avec notre CV.
2. E03 : passer les `_cat` en `categorical_feature` LightGBM, ou les encoder (fréquence / cible dans les folds), une fois que l'audit a validé la v1.
3. Comparatifs : XGBoost GPU (`device="cuda"`) et CatBoost GPU sur la même v0/v1, puis une moyenne de rangs sur les OOF si les scores sont proches.
4. Optuna seulement une fois les données stabilisées (budget annoncé ~20 min, n_jobs ≤ 16).

## Fichiers produits
- `usine/src/utils.py`, `usine/src/03_train.py`, `usine/src/04_soumission.py`
- `usine/experiences/folds.csv`
- `usine/experiences/registre.csv` (lignes E00, E01)
- `usine/experiences/oof/E00_oof.csv`, `E00_test.csv`, `E00_folds_detail.csv`
- `usine/experiences/oof/E01_oof.csv`, `E01_test.csv`, `E01_folds_detail.csv`
- `usine/submissions/2_302.csv`
- `usine/logs/03_train_E00_E01.log`, `usine/logs/04_sub_2_302.log`
