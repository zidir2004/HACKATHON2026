# Reviewer — revue des itérations E02→E16 et de la soumission 2_303.csv (E09) (2026-11-04)

## Verdict : VALIDÉ AVEC RÉSERVES — soumission `2_303.csv` DÉPOSABLE

Aucun point bloquant. Format conforme (contrôle indépendant), ordre identique au mélange E09, pas de fuite trouvée, gain du mélange réel sur le découpage 42 (5/5 folds, y compris contre le meilleur membre de chaque fold). Réserves : robustesse sur une autre graine de découpage pas encore disponible au moment du rapport (job lancé, voir §5), E02 « garde » statistiquement fragile, registre à fins de ligne mixtes.

## Ce qui a été fait
1. Lecture de `src/03_train.py`, `06_melange.py`, `07_rangs_vers_proba.py`, `04_soumission.py`, `utils.py`, du rapport `06_ml-engineer_iterations.md`, du registre, de `comparaisons.csv`, des fichiers job (`queue/*.done`) et des logs.
2. Contrôle de format de `2_303.csv` par un script indépendant (Python stdlib, sans `verifier_soumission`) + Spearman et inversions vs `oof/E09_test.csv`.
3. Recalcul indépendant (AUC par rangs, stdlib) du Gini par fold de E02, E07, E08, E09 depuis les OOF ; contrôle folds OOF = `folds.csv` ; recalcul du mélange E09 à partir des rangs des 3 membres.
4. Relecture du registre avec le module `csv`.
5. Job `91_revue_E09_s42` et `91_revue_E09_s7` (`src/91_revue_E09.py`, n'écrit ni registre ni OOF) : E02/E07/E08 + mélange avec **nombre d'arbres fixe** (600/400/850, sans early stopping sur le fold rapporté), découpage 42 et découpage graine 7.

## 1. Format de `2_303.csv` — CONFORME
| Contrôle | Résultat |
|---|---|
| Lignes | 148 803 = test.csv |
| En-tête | absent (1re ligne `7,0.04698824`) |
| Colonnes | 2, virgule, 0 ligne mal formée, pas de BOM |
| Ids | identiques à test.csv, même ordre, uniques, croissants |
| Valeurs | 0 NaN, 0 hors [0,1] ; min 0,00822, max 0,45873, moyenne 0,03635 (taux train 0,03645) |
| Fins de ligne | LF uniquement (0 CR), fichier terminé par LF |
| Ordre vs `E09_test.csv` | Spearman = 0,99999999998 ; **0 inversion** dans l'ordre du mélange. Les 2 323 ex-aequo de 2_303 viennent de l'arrondi à 8 décimales (E09_test en a 15 162 par moyennes de rangs) : sans effet mesurable sur le Gini. |

Le remappage `07_rangs_vers_proba.py` est correct : `pred[argsort(mélange)] = sort(moyenne des probas)` conserve l'ordre (mergesort stable), valeurs dans l'échelle des probabilités. Pour le Gini (seul l'ordre compte), le fichier est équivalent à `_2_303_rangs.csv`. Remarque : ces valeurs ne sont pas des probabilités calibrées individuellement (c'est la distribution des probas moyennes réattribuée) ; sans importance pour la métrique Gini.

## 2. Fuites — aucune trouvée
- **E05 (encodage cible)** : `encodage_cible()` est appelée dans la boucle de folds avec `X.loc[tr]`, `y[tr]` uniquement ; lignes d'entraînement encodées par K-fold interne (5, stratifié) ; validation et test encodés avec la table apprise sur tout le fold d'entraînement ; lissage m = 100 vers le prior du fold. Correct.
- **Membres du mélange** : les 3 OOF (E02, E07, E08) ont exactement les folds de `folds.csv` (contrôlé ligne à ligne), aucune OOF nulle (chaque ligne prédite une fois, hors fold). E07/E08 lancés avec `--base E02 --base-version v1` (job `07_train_E07_E08`), donc mêmes données v1.
- **Prédiction test** : `pred_test += p_te / 5` dans la boucle → moyenne des 5 modèles de fold. OK.
- Le `+1` sur les catégorielles (CatBoost / cat natives) est une transformation fixe, non apprise : pas de fuite.
- Biais connu et commun à tous les membres (déjà signalé en revue E01) : early stopping / `use_best_model` sur le fold rapporté → +0,0005 à +0,0008 optimiste par modèle.

## 3. Le gain du mélange est-il réel ?
Gini par fold recalculé depuis les OOF (identique au registre) :

| Exp | f0 | f1 | f2 | f3 | f4 | moyenne |
|---|---|---|---|---|---|---|
| E02 LGB | 0,27775 | 0,26778 | 0,29003 | 0,28594 | 0,28289 | 0,28088 |
| E07 XGB | 0,27832 | 0,26611 | 0,28862 | 0,28505 | 0,28556 | 0,28073 |
| E08 CB | 0,27849 | 0,27001 | 0,28981 | 0,28770 | 0,28153 | 0,28151 |
| **E09** | 0,28079 | 0,27007 | 0,29180 | 0,28847 | 0,28600 | **0,28343** |

- Mélange recalculé à partir des rangs des membres : écart max 2e-16 avec `E09_oof.csv`.
- Δ apparié vs E02 : +0,00255, 5/5 folds, écart-type des Δ 0,0005 (t ≈ 10 sur 5 folds). **E09 bat aussi le meilleur membre de chaque fold sur 5/5** (marges +0,0023, +0,0001, +0,0018, +0,0008, +0,0004) : le gain n'est pas un artefact du choix de la référence.
- Biais de sélection : faible. Poids égaux (aucun paramètre appris) ; les membres sont les 3 familles prévues d'avance avec réglages par défaut, pas choisis parmi beaucoup de candidats ; E14 (légèrement meilleur sur l'OOF) n'a pas été retenu, ce qui est prudent. Seule sélection : 2 mélanges essayés, effet négligeable.
- **Robustesse autre graine / arbres fixes** : voir §5.

## 4. Registre
- `registre.csv` se relit correctement : 13 colonnes sur toutes les lignes (module csv), les champs « changement » contenant des virgules sont bien entre guillemets. Pas de casse.
- **MINEUR** : fins de ligne mixtes (17 lignes en CRLF écrites par `csv.DictWriter` sans `lineterminator`, en-tête en LF) ; ordre des lignes non chronologique par id (E07 avant E05/E06, normal car jobs parallèles).
- **MINEUR** : la colonne `delta_vs_meilleur` est calculée vs le meilleur score déjà au registre au moment de l'écriture (ex. E09 +0,00164 = vs E13, E16 −0,00254 vs E09), pas vs la référence : à renommer ou documenter, le Δ apparié vs réf est dans `commentaire` et `comparaisons.csv`.

## 5. Robustesse (job en cours au moment du rapport)
Jobs `91_revue_E09_s42` et `91_revue_E09_s7` (arbres fixes, sans early stopping). Les deux jobs en parallèle sont lents (~5,5 min par fold, CatBoost GPU partagé) : résultat complet attendu ~25 min après lancement dans `usine/logs/91_revue_E09_s42.log` et `..._s7.log` (lignes « melange CV » et « melange - lgb »).
Fold 0 disponible : graine 42 lgb 0,27735 / xgb 0,27750 / cb 0,27888 (cohérent avec E02/E07/E08 avec early stopping : écarts −0,0004 à +0,0004) ; graine 7 lgb 0,26755 / xgb 0,26614 / cb 0,26690. **Non vérifié à ce stade : le gain du mélange sur la graine 7.** Ce point ne bloque pas le dépôt (le mélange ne peut presque pas faire pire que ses membres équivalents, et le dépôt donnera la mesure officielle), mais il conditionne le chiffre annoncé.

## 6. E02 « garde » (retrait des ps_calc_*) — défendable, mais pas comme un gain prouvé
- Δ apparié +0,00218, 4/5 folds, mais écart-type des Δ 0,00311 (fold 3 : −0,0024, fold 4 : +0,0072) → t ≈ 1,4 sur 5 folds : **non significatif**.
- Défendable pour d'autres raisons, convergentes : audit data engineer (calc seules Gini 0,0021, aucune variable calc informative), SHAP analyste (8 % d'importance répartie sur du bruit), et simplification (37 variables au lieu de 57). À formuler : « retrait sans perte, simplification » plutôt que « +0,002 ».

## Constats classés
| Niveau | Constat | Preuve | Correction |
|---|---|---|---|
| IMPORTANT | Gain du mélange non encore confirmé sur une 2e graine de découpage | job 91 en cours (§5) | lire `logs/91_revue_E09_s7.log` ; annoncer ~0,281–0,283 tant que non confirmé |
| IMPORTANT | E02 présenté comme gain alors que Δ non significatif (std Δ 0,0031) | `comparaisons.csv` ligne E02 | reformuler en « retrait neutre à positif, simplification » ; la règle de verdict devrait exiger Δ moy > std(Δ)/√5 ×2 ou 5/5 |
| MINEUR | Registre en fins de ligne mixtes CRLF/LF | 17 CR sur 18 lignes | `lineterminator="\n"` dans `ajouter_registre` |
| MINEUR | `delta_vs_meilleur` ≠ Δ vs réf, source de confusion | E09 +0,00164 vs E13 | renommer `delta_vs_meilleur_registre` ou documenter |
| MINEUR | 2_303 = probas moyennes réattribuées, non calibrées individuellement | `07_rangs_vers_proba.py` | sans effet sur le Gini ; le dire dans la soutenance si on parle de probabilités |
| INFO | Biais early stopping commun à tous les membres (cf. revue E01) | revue 90 | inchangé : retirer ~0,0005–0,001 au chiffre annoncé |

## Ce que je n'ai pas pu vérifier
- Reproduction exacte de E07/E08 (GPU) : non relancés à l'identique ; seuls les arbres fixes du job 91 donnent des chiffres voisins.
- Résultat complet du contrôle de robustesse graine 7 (en cours, voir §5).
