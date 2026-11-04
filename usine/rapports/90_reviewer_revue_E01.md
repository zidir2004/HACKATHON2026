# Reviewer — revue de la baseline E01 et de la soumission 2_302.csv (2026-11-04 09:35)

## Verdict : VALIDÉ AVEC RÉSERVES — soumission `2_302.csv` DÉPOSABLE

Aucun point bloquant. Les réserves portent sur l'interprétation du score (biais faible de l'early stopping, sensibilité au découpage) et sur un détail de format (fins de ligne CRLF) qui n'empêche pas le dépôt.

## Mission reçue
Revue indépendante, avant dépôt, de la baseline E01 (LightGBM v0) et de la soumission `usine/submissions/2_302.csv` : format, fuites, reproductibilité, cohérence des scores.

## Ce qui a été fait
1. Lecture de `usine/src/utils.py`, `03_train.py`, `04_soumission.py`, du rapport ML engineer et des messages de l'audit.
2. **Contrôle de format par un script indépendant** (Python stdlib, module csv, sans réutiliser `verifier_soumission`) sur `2_302.csv`, `test.csv` et `submission_example.csv`.
3. **Recalcul indépendant du Gini** (AUC par rangs, stdlib) depuis `usine/experiences/oof/E01_oof.csv`, global et par fold ; contrôle folds OOF = `folds.csv` ; Gini de l'id vers la cible ; recouvrement train/test.
4. **Job `90_revue_E01`** (`usine/src/90_revue_E01.py`, n'écrit ni registre ni OOF) : a) réexécution exacte de E01 ; b) même modèle avec 600 arbres fixes (sans early stopping) ; c) E01 sur 2 autres découpages (graines 7 et 2026), avec et sans early stopping.

## Résultats clés

### Format de `2_302.csv` (contrôle indépendant)
| Contrôle | Résultat |
|---|---|
| Lignes | 148 803 = lignes de test.csv (hors en-tête) |
| En-tête | absent (1re ligne `7,0.04736622`) |
| Colonnes / séparateur | 2 colonnes, virgule, 0 ligne mal formée, 0 guillemet, 0 point-virgule, pas de BOM |
| Ids | identiques à test.csv, même ordre, strictement croissants, uniques |
| Valeurs | 0 NaN, 0 vide, 0 hors [0,1] ; min 0,00871, max 0,38579, moyenne 0,03641 (taux train 0,03645), 146 511 valeurs distinctes |
| Fidélité | max \|soumission − E01_test.csv\| = 5e-9 (arrondi à 8 décimales) |
| Fins de ligne | **CRLF (`\r\n`)** sur les 148 803 lignes, alors que `submission_example.csv` est en LF ; fichier terminé par un saut de ligne |

### Cohérence des scores
- Gini OOF recalculé : **0,27856** (registre : 0,27856). Gini par fold : 0,27657 / 0,26470 / 0,28818 / 0,28836 / 0,27568, moyenne **0,27870**, écart-type (ddof=0) **0,00886** : identique au registre.
- Folds de l'OOF = `folds.csv` (446 409 ids) ; ~3 254 positifs par fold.
- Distribution test vs OOF : quantiles 1/10/25/50/75/90/99 % test = 0,0132 / 0,0186 / 0,0240 / 0,0319 / 0,0433 / 0,0592 / 0,1057 ; OOF = 0,0129 / 0,0185 / 0,0238 / 0,0318 / 0,0433 / 0,0596 / 0,1059. Quasi identiques : pas de signe de décalage ni d'erreur d'alignement. (Le max test plus bas, 0,386 vs 0,549, s'explique par la moyenne de 5 modèles qui lisse les extrêmes.)

### Fuites
- `liste_features` exclut `id` et `target` (vérifié aussi par assertion dans le job : 57 variables, ni id ni target).
- Gini(id → cible) = 0,00072 ; 0 id commun train/test ; aucune transformation apprise en v0 (LightGBM brut) ; pour E00, le StandardScaler est dans le pipeline donc ajusté dans le fold.
- Pas de seuil (Gini = rangs) : rien à optimiser de ce côté.

### Early stopping et stabilité (job 90_revue_E01)
| Découpage | Early stopping (fold de validation) | 600 arbres fixes | Écart |
|---|---|---|---|
| graine 42 (protocole) | 0,27870 ± 0,00886 (OOF 0,27856) | 0,27820 ± 0,00882 (OOF 0,27810) | +0,0005 |
| graine 7 | 0,27624 ± 0,00660 | 0,27557 ± 0,00641 | +0,0007 |
| graine 2026 | 0,27796 ± 0,00312 | 0,27718 ± 0,00329 | +0,0008 |

- **Reproductibilité** : la réexécution (a) redonne exactement 0,278699 / 0,278561 et les mêmes Gini par fold.
- **Biais de l'early stopping** : +0,0005 à +0,0008 de Gini, soit ~0,1 écart-type entre folds. Faible, mais systématique (même signe sur les 3 découpages).
- **Sensibilité au découpage** : le Gini CV moyen varie de 0,2762 à 0,2787 selon la graine (amplitude 0,0025) ; le découpage du protocole (42) est le plus favorable des trois. Score « honnête » attendu ≈ **0,277**.

## Décisions et justification
- **Format conforme → déposable.** Tous les points stricts de l'énoncé sont respectés. Les fins de ligne CRLF viennent de `pandas.to_csv` sous Windows ; les lecteurs usuels (pandas, module csv, `float()` qui ignore `\r`) les lisent sans problème. Risque faible, donc MINEUR, mais correction triviale recommandée pour coller à l'exemple.
- **Early stopping sur le fold de validation : acceptable pour comparer des expériences**, car le biais (~+0,0006) est le même pour toutes les expériences LightGBM et 10 fois plus petit que l'écart-type. Il doit être **signalé** dans les rapports et l'oral, et il pénalise la comparaison avec un modèle sans early stopping (ex. logistique E00, ou un modèle à nombre d'arbres fixe).
- **Code** : lisible, une config par expérience, graines fixées (folds 42, LightGBM `seed=42`), folds relus depuis `folds.csv` avec assertion d'alignement. Reproductible au chiffre près (vérifié).

## Points d'attention
| Gravité | Constat | Preuve | Correction proposée |
|---|---|---|---|
| IMPORTANT | Score CV légèrement optimiste : early stopping sur le fold rapporté + découpage 42 favorable | +0,0005 (graine 42) à +0,0008 ; Gini 0,2762–0,2787 selon la graine | Annoncer « ≈ 0,277–0,279 » plutôt que 0,2787 ; ne pas interpréter l'écart CV/leaderboard < 0,003 ; pour l'oral, mentionner le biais |
| IMPORTANT | La règle de verdict `gain > écart-type` compare à l'écart-type entre folds, pas à l'écart-type de la **différence appariée** par fold | `utils.ajouter_registre` | Ajouter au registre l'écart moyen fold à fold vs la référence et son écart-type (mêmes folds → différence appariée beaucoup plus précise) ; pour les gains < 0,005, confirmer sur une 2e graine de découpage |
| MINEUR | Fins de ligne CRLF dans la soumission, LF dans l'exemple | od -c : `\r\n` sur 148 803 lignes | `sub.to_csv(..., lineterminator="\n")` dans `utils.ecrire_soumission` pour les prochaines soumissions |
| MINEUR | Écart-type des folds calculé en ddof=0 (`np.std`) | `03_train.py` | Rien à changer, mais l'écrire (ddof=1 donnerait 0,0099) |
| MINEUR | Le max des prédictions test (0,386) < max OOF (0,549) | quantiles ci-dessus | Normal (moyenne de 5 modèles) ; sans effet sur le Gini |

Non vérifié : le Gini officiel (leaderboard) n'est pas encore connu ; l'écart CV/officiel sera à contrôler après le dépôt.

## Recommandations
1. Déposer `2_302.csv` tel quel ; noter le score officiel dans le registre et le comparer à **0,277–0,279**.
2. Passer `lineterminator="\n"` dans `ecrire_soumission` avant la prochaine soumission.
3. Ajouter au registre la différence appariée par fold vs la référence (moyenne ± écart-type) pour juger les gains de 0,002–0,005 (retrait des `ps_calc_*`, encodages).
4. Pour une soumission finale, envisager de fixer le nombre d'arbres (médiane des folds) ou un early stopping sur un sous-ensemble interne du fold d'entraînement, pour supprimer le biais.

## Fichiers produits
- `usine/rapports/90_reviewer_revue_E01.md` (ce rapport)
- `usine/src/90_revue_E01.py` (contrôle reproductibilité / early stopping / graines ; n'écrit ni registre ni OOF)
- `usine/logs/90_revue_E01.log`, `usine/logs/90_revue_E01.status` (0)
- `usine/queue/90_revue_E01.job` → .done
