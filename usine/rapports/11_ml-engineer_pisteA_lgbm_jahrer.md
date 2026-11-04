# 11 - ML engineer, piste A : LightGBM de la solution 1re place (E29-E37)

Source : solution 1re place Kaggle Porto Seguro, M. Jahrer (capture fournie par l'utilisateur). Il mélange
1 LightGBM et 5 réseaux, avec les mêmes variables pour tous : ps_calc_* retirées et one-hot des *_cat.
Paramètres LightGBM : binary, 1400 arbres, lr 0,01, max_bin 255, num_leaves 31, min_data_in_leaf 1500,
feature_fraction 0,7, bagging 0,7 (freq 1), lambda_l1 1, lambda_l2 1. Son CV 5 folds : Gini 0,28843 sur 595 212 lignes.

## Hypothèse
Les paramètres publiés, très régularisés (min_data_in_leaf 1500), et le one-hot des _cat donnent un LightGBM
au moins aussi bon que E13 et surtout **différent** de nos arbres, ce qui doit être utile dans le mélange.

## Protocole
- Mêmes folds (`usine/experiences/folds.csv`), données v1 (37 var., sans calc), CPU uniquement (n_jobs 8, 2-3 jobs en parallèle).
- **Nombre d'arbres fixe** (1400, puis 1050 pour E35), donc **pas d'early stopping** : le Gini d'un fold n'est pas
  optimiste. La courbe AUC de validation est seulement enregistrée pour diagnostic (`diag_iter_max_auc`
  dans `oof/E29_folds_detail.csv`) et ne sert à rien choisir.
- One-hot : non supervisé, colonne par colonne, modalités de train ∪ test, -1 = modalité à part, soit 184 indicatrices
  (`03_train.one_hot`). Pas besoin de le faire dans les folds.
- Règle : `garde` si Δ>0 et (5/5 ou Δ > 2·sd(Δ)/√5). Tout `garde` d'un modèle seul est confirmé par la moyenne de 3 graines (42/7/2026, `09_verif_graines.py`).
- Code : `usine/src/03_train.py` (LGB_JAHRER, configs E29/E30/E30b/E35, `one_hot`, arbres fixes dans `ajuster_lightgbm`),
  `06_melange.py`, `09_verif_graines.py`, `07_rangs_vers_proba.py`. Jobs 40 à 49.

## Résultats (Gini CV moyen ± écart-type entre folds, Δ apparié par fold)

| Exp | Changement | Gini CV | Réf | Δ moy | Folds | Verdict |
|---|---|---|---|---|---|---|
| E13 | réf. notre meilleur LightGBM seul | 0,28179 ± 0,0077 | - | - | - | référence |
| E29 | paramètres Jahrer, 1400 arbres, _cat numériques | 0,27880 ± 0,0073 | E13 | −0,00299 | 0/5 | **rejeté** |
| E30 | E29 + one-hot, _cat d'origine retirées | 0,27984 ± 0,0062 | E29 | +0,00104 | 4/5 | neutre |
| E30b | E29 + one-hot, _cat d'origine **gardées** | 0,27991 ± 0,0077 | E29 | +0,00112 | 5/5 | garde |
| E32 | E30b vs E29, **3 graines** de chaque côté | 0,28052 ± 0,0077 | E29m | +0,00108 | 4/5 (seuil 0,00099) | **garde confirmé** |
| E33 | E30 vs E29, 3 graines | 0,28069 ± 0,0067 | E29m | +0,00125 | 4/5 (seuil 0,00135) | neutre |
| E35 | E30b avec 1050 arbres (1400 × 357k/476k, fixé a priori) | 0,28169 ± 0,0077 | E30b | +0,00177 | 5/5 | garde |
| E36 | E35 vs E30b, 3 graines | 0,28210 ± 0,0075 | E32 | +0,00158 | 5/5 | **garde confirmé** |
| E31 | rangs égaux E30b + E07 + E08 (E30b remplace E02) | 0,28418 ± 0,0077 | E09 | +0,00075 | 5/5 | garde |
| E31c | idem avec E30 | 0,28425 ± 0,0071 | E09 | +0,00083 | 5/5 | garde |
| E31b | E02 + E30b + E07 + E08 (4 membres) | 0,28390 ± 0,0077 | E09 | +0,00048 | 5/5 | garde (moins bon que E31) |
| **E34** | rangs égaux **E32** (E30b, 3 graines) + E07 + E08 | **0,28426 ± 0,0076** (OOF 0,28414) | E17 | +0,00055 | 5/5 | **garde** (vs E09 : +0,00083, 5/5) |
| E37 | rangs égaux E36 (E35, 3 graines) + E07 + E08 | 0,28426 ± 0,0076 | E34 | +0,00000 | 3/5 | neutre |

Pas de seuil (Gini). Temps : ~95-125 s par LightGBM 1400 arbres (n_jobs 8), ~160 s pour 1050 arbres avec 3 jobs en parallèle,
quelques secondes par mélange. Environ 35 min au total.

## Ce qui se reproduit, et ce qui ne se reproduit pas
- **Le score de Jahrer (0,28843) ne se reproduit pas** : E29 donne 0,27880, et 0,28052 avec le one-hot et 3 graines.
  La comparaison n'est pas directe : nous n'avons que 446 409 lignes (357k par fold d'entraînement contre 476k)
  et un autre découpage. Sur NOS folds, ses paramètres font moins bien que notre E13 (−0,0030, 0/5).
- **Cause probable : 1400 arbres, c'est trop pour notre taille de train.** Le pic d'AUC de validation (diagnostic seul)
  se situe entre 535 et 1088 arbres selon le fold. En ramenant a priori à 1050 arbres (proportionnellement aux lignes,
  sans regarder la validation), E35/E36 gagnent +0,0016 (5/5, 3 graines). Le modèle rejoint alors notre LightGBM réglé :
  E36 vs E17a = −0,00007 (3/5), et E35 vs E13 = −0,0001 (2/5), soit une égalité.
  Honnêteté : la règle 357/476 est fixée a priori, mais j'ai décidé de tester E35 après avoir vu le diagnostic.
  Le risque de biais reste faible (un seul essai, aucune grille), et E34, le mélange retenu, n'utilise pas E35.
- **Le one-hot des _cat se reproduit, mais avec un petit effet** : +0,0011 (confirmé sur 3 graines), à condition de **garder**
  les colonnes _cat d'origine (E30b). Sans elles (E30, sa variante exacte), la moyenne est proche mais l'effet est plus bruité (neutre).
- **La vraie valeur est dans le mélange.** Le LightGBM Jahrer avec one-hot est moins corrélé à XGBoost que notre E02
  (corrélation des rangs OOF 0,968 contre 0,987). Remplacer E02 par lui dans le mélange donne +0,0008 vs E09 et
  +0,00055 vs E17 sur 5/5 folds, avec 3 graines. C'est cohérent avec son choix de mêler des modèles différents à poids égaux.
- E37 (version 1050 arbres) = E34 : la meilleure qualité du membre seul ne s'ajoute pas à la diversité qu'il apportait déjà.

## Livrables
- Prédictions OOF + test pour la piste B : `usine/experiences/oof/{E29,E30,E30b,E32,E33,E35,E36,E34,E37}_oof.csv` / `_test.csv`
  (+ graines `E29s7`, `E30bs7`, ... ; `E29m` = moyenne 3 graines de E29). Ingrédient LGB recommandé : **E32** ou **E36** (3 graines).
- **Soumission candidate `usine/submissions/2_305.csv` = E34** (pas déclarée déposable). Auto-contrôle : 148 803 lignes, sans en-tête,
  ids identiques et dans le même ordre que 2_304, 0 NaN, LF, moyenne 0,0364 (taux train 3,64 %), Spearman sortie/mélange 1,000000,
  Spearman vs 2_304 0,9982.

## Verdict et prochaine idée
Nouvelle référence de mélange : **E34 (0,28426 ± 0,0076, OOF 0,28414)**, +0,00055 vs E17 (5/5). Les gains restent faibles
devant l'écart entre folds (0,0076) : annoncer environ 0,282-0,284.
Prochaine idée : moyenne simple des rangs E34 + réseaux de la piste B (DAE), dès que leurs OOF existent sur nos folds.
C'est là que Jahrer tirait l'essentiel de son gain.
