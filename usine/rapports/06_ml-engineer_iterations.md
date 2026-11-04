# ML engineer — boucle d'expériences E02 → E16 (2026-11-04, 09:40–10:05)

## Hypothèse et méthode
Améliorer E01 (LightGBM v0, Gini CV 0,27870 ± 0,00886) par petits pas : d'abord les données (v1/v2/v3, encodages des catégorielles), puis d'autres modèles (XGBoost, CatBoost sur GPU), puis un mélange.
- **Un seul changement par expérience**, toujours comparé à la meilleure configuration retenue (colonne « réf »).
- Mêmes folds pour tout (`usine/experiences/folds.csv`). Encodage cible calculé **dans** les folds d'entraînement, avec un K-fold interne (5) pour les lignes d'entraînement elles-mêmes : pas de fuite.
- **Comparaison appariée** (demande du reviewer) : Gini_fold(exp) − Gini_fold(réf) sur chacun des 5 folds → Δ moyen et nombre de folds gagnés. Détail : `usine/experiences/comparaisons.csv`, repris dans le commentaire du registre.
- Règle de verdict : `garde` si Δ > 0 ET (≥ 4/5 folds gagnés OU Δ > std de la réf) ; `rejete` si Δ < 0 ET (≤ 1/5 folds gagnés OU perte > std) ; sinon `neutre`.
- Code : `usine/src/03_train.py` (configs E02–E16, préparation dans les folds, modèles LightGBM/XGBoost/CatBoost, comparaison appariée), `usine/src/06_melange.py` (mélange par moyenne des rangs). `utils.ecrire_soumission` écrit maintenant en LF.

## Tableau récapitulatif
| Exp | Changement (vs réf) | Réf | Gini CV ± std | Δ apparié moyen | Folds gagnés | Verdict |
|---|---|---|---|---|---|---|
| E02 | v1 : retrait des 20 ps_calc_* | E01 | 0,28088 ± 0,00768 | +0,00218 | 4/5 | **garde** (et 20 variables en moins) |
| E03 | v2 : + nb_manquants + 5 indicateurs *_manq | E02 | 0,28001 ± 0,00815 | −0,00087 | 2/5 | neutre (non retenu : ajoute 6 variables) |
| E04 | v3 : sans ps_ind_14, one-hot regroupés | E03 | 0,28060 ± 0,00833 | +0,00059 | 3/5 | neutre |
| E04 | (même run, comparé à la meilleure) | E02 | — | −0,00028 | 2/5 | neutre (non retenu) |
| E05 | + encodage cible lissé de ps_car_11_cat (dans les folds) | E02 | 0,28115 ± 0,00714 | +0,00027 | 2/5 | neutre (non retenu) |
| E06 | toutes les _cat en catégorielles natives LightGBM | E02 | 0,27100 ± 0,00766 | −0,00988 | 0/5 | **rejete** |
| E07 | XGBoost GPU (depth 5, eta 0,03) sur v1 | E02 | 0,28073 ± 0,00804 | −0,00015 | 2/5 | neutre (équivalent, utile au mélange) |
| E08 | CatBoost GPU (depth 6, lr 0,05, cat natives) sur v1 | E02 | 0,28151 ± 0,00705 | +0,00063 | 3/5 | neutre (meilleur modèle seul, utile au mélange) |
| **E09** | **mélange rangs, poids égaux E02 + E07 + E08** | E02 | **0,28343 ± 0,00758** | **+0,00255** | **5/5** | **garde** |
| E10 | LightGBM feature_fraction 0,6 → 0,4 | E02 | 0,28100 ± 0,00795 | +0,00012 | 2/5 | neutre |
| E11 | LightGBM num_leaves 24 → 16 | E02 | 0,28049 ± 0,00777 | −0,00039 | 1/5 | rejete (perte minime) |
| E12 | LightGBM min_child_samples 150 → 500 | E02 | 0,28041 ± 0,00730 | −0,00047 | 1/5 | rejete (perte minime) |
| E13 | LightGBM learning_rate 0,02 → 0,01 | E02 | 0,28179 ± 0,00769 | +0,00091 | 5/5 | garde (modèle seul) |
| E14 | mélange avec E13 au lieu d'E02 | E09 | 0,28368 ± 0,00763 | +0,00025 | 3/5 | neutre (non retenu) |
| E15 | CatBoost lr 0,05 → 0,03 | E08 | 0,28115 ± 0,00719 | −0,00035 | 2/5 | neutre (non retenu) |
| E16 | XGBoost eta 0,03 → 0,015 | E07 | 0,28114 ± 0,00780 | +0,00041 | 3/5 | neutre (non retenu) |

Seuil : sans objet (le Gini ne dépend que de l'ordre des prédictions). E09 n'a pas été listée dans l'ordre de la mission (E09 avant E10) : les E10–E16 sont des pistes ajoutées ensuite.

## Interprétation (en français simple)
- **E02 (sans ps_calc)** : les 20 variables calc sont bien du bruit ; les retirer fait gagner un peu (4 folds sur 5) et simplifie le modèle (37 variables au lieu de 57). Nouvelle base.
- **E03 (indicateurs de manquants)** : LightGBM voit déjà les -1 comme une valeur à part ; les indicateurs n'apportent rien de plus (légère perte, 2/5).
- **E04 (v3)** : regrouper les one-hot et retirer ps_ind_14 ne change rien de mesurable. Limite : v3 contient aussi les indicateurs de v2, donc le test « v1 + regroupements seuls » n'a pas été fait ; il y a peu à en attendre.
- **E05 (encodage cible ps_car_11_cat)** : gain moyen +0,0003 seulement, 2/5 folds. Les arbres découpent déjà assez finement ps_car_11_cat lue comme un nombre (104 modalités, toutes > 850 assurés). Non retenu.
- **E06 (catégorielles natives)** : nette perte (−0,010, 0/5). Les découpages « par groupe de modalités » de LightGBM surapprennent ici malgré la régularisation ; garder les _cat en numérique pour LightGBM.
- **E07 / E08 (XGBoost, CatBoost GPU)** : même niveau que LightGBM (CatBoost un peu au-dessus, 0,2815), mais leurs prédictions diffèrent (corrélation des rangs 0,96 entre CatBoost et LightGBM) : c'est ce qui rend le mélange utile.
- **E09 (mélange)** : +0,0026 et **les 5 folds gagnés**, le gain le plus solide de la série. Poids égaux, rien n'est appris sur l'OOF. Pour info, les poids « optimaux » trouvés sur l'OOF (0,2/0,3/0,5) ne donnent que 0,28352 en OOF contre 0,28328 : ils n'ont pas été retenus (biais optimiste).
- **E10–E16 (réglages)** : LightGBM est déjà sur un plateau ; seul lr 0,01 (E13) gagne un peu sur les 5 folds (+0,0009), mais une fois dans le mélange (E14) le gain tombe à +0,0003 (3/5). Baisser le pas de CatBoost ou d'XGBoost n'apporte rien.

## Configuration retenue et soumission candidate
- **E09** = moyenne des rangs (poids égaux) de LightGBM v1 (E02), XGBoost GPU v1 (E07), CatBoost GPU v1 (E08). Gini CV **0,28343 ± 0,00758**, OOF 0,28328.
- À annoncer avec la correction du reviewer (early stopping sur le fold rapporté, découpage 42 favorable) : environ **0,281–0,283** attendu.
- `usine/submissions/2_303.csv` (job `12_sub_2_303`) : 148 803 lignes, sans en-tête, ids identiques et dans l'ordre de `test.csv`, aucun NaN, valeurs dans [0,1], fins de ligne LF (0 CR). Corrélation de Spearman avec 2_302 : 0,990.
- **Attention** : les valeurs sont des **rangs normalisés** (moyenne 0,5), pas des probabilités (le taux de positifs du train est 0,036). Sans effet sur le Gini (seul l'ordre compte), mais à valider par le reviewer si le format attend des probabilités. **Non déclarée déposable** : contrôle du reviewer attendu.
- Prédiction test = moyenne des 5 modèles de fold de chaque membre (pas de ré-entraînement sur tout le train).

## Temps de calcul
LightGBM ~30–40 s par expérience (5 folds, 16 threads), lr 0,01 : 100 s ; XGBoost GPU : 20–30 s ; CatBoost GPU : 6,5 min (lr 0,05) à 7,5 min (lr 0,03) ; mélange : quelques secondes. Total machine ~25 min, jobs CPU et GPU en parallèle.

## Prochaines idées recommandées
1. Moyenne de plusieurs graines (3–5) pour chaque membre du mélange : réduit la variance, gain attendu faible mais fiable.
2. Ajouter un 4e membre différent (ex. LightGBM v1 + encodage cible ps_car_11_cat, ou régression logistique sur variables encodées) pour diversifier le mélange.
3. Garder v1 comme version de données de référence ; ne plus investir dans les indicateurs de manquants ni dans les catégorielles natives LightGBM.
