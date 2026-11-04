# 94 — Revue de `usine/submissions/2_308.csv` (refit 100 % des 4 membres de 2_306)

**Verdict : DÉPOSABLE** — VALIDÉ AVEC RÉSERVES (aucun bug ni fuite ; gain non mesurable en CV ; 1 réserve IMPORTANTE sur E80).

## 1. Format (script indépendant, Python stdlib, fichier contrôlé tel quel)
- 148 803 lignes, 2 colonnes, **sans en-tête** (1re ligne `7,0.04362918`), séparateur virgule, LF uniquement, LF final.
- ids **identiques et dans le même ordre** que `test.csv` ; 0 doublon ; 0 NaN ; min 0,00712, max 0,43647 (dans [0,1]).
- Moyenne 0,03674 (taux train 0,03645) : plausible. → **OK**

## 2. Cohérence des prédictions (recalcul indépendant)
- Spearman 2_308 vs 2_306 = **0,99815** (annoncé 0,9982 : confirmé).
- Spearman vs 2_308 / vs 2_306 : E80 0,9796/0,9752 ; E81 0,9859/0,9834 ; E82 0,9804/0,9792 ; E83 0,9639/0,9648.
  Les versions par folds E30/E07/E08/E32c : 0,9839/0,983 ; 0,9868/0,9891 ; 0,9808/0,9846 ; 0,9626/0,9645. Aucun membre aberrant.
- Quantiles (0 / 1 / 10 / 25 / 50 / 75 / 90 / 99 / 99,9 / 100 %) :
  2_308 : 0,0071 0,0119 0,0178 0,0235 0,0320 0,0441 0,0610 0,1098 0,1644 0,4365
  2_306 : 0,0075 0,0123 0,0182 0,0239 0,0322 0,0441 0,0605 0,1076 0,1585 0,3946
  Distribution un peu plus étalée aux extrêmes : attendu (un modèle unique au lieu d'une moyenne de 5 modèles de fold). Sans effet sur le Gini (rangs).

## 3. Lecture du code `17_refit_complet.py` / `18_combiner.py`
- Données : `preparer()` recopie 03_train (v1, +1 sur `_cat` pour CatBoost, one-hot « remplace » pour E30 : 207 variables, vu dans le log). `assert not te_cols` : pas d'encodage cible. La cible n'est jamais une variable (`liste_features`). OK.
- Paramètres : `T.CONFIGS[exp_cv]["params"]` réutilisés tels quels ; seul le nombre d'arbres change. Pas d'early stopping (impossible sans validation) : correct. CatBoost sans `od_*` : correct.
- Réseau E83 : DAE `dae.pt` gelé (non supervisé, déjà utilisé par 2_306), réseau réentraîné sur `i_tr` = 100 % du train, test jamais dans les mini-lots supervisés ; `predire()` passe en `eval()` et `train()` est remis à chaque époque. OK.
- Graines : SEED=42 dans les params LGB/XGB/CB ; `torch.manual_seed` par graine réseau. Reproductible au non-déterminisme GPU près (XGB/CB/réseau sur GPU).
- Combineur : rangs à poids égaux, quantile mapping, réindexation sur l'ordre de test.csv, contrôles format/LF. OK.
- **Aucune fuite identifiée.** Pas de jeu test dans l'entraînement supervisé.

## 4. Facteur × 1,2 sur le nombre d'arbres
- XGBoost / CatBoost (moy. best_iteration × 1,2 ≈ ratio des tailles 1,25) : règle standard, risque faible. **OK.**
- Réseau : 6 époques fixes, 1,25× plus de pas par époque, moyenne des époques : risque faible. **OK.**
- **IMPORTANT — E80 (LightGBM E30)** : 1400 arbres n'était pas un best_iteration mais un nombre fixe déjà **au-delà de l'optimum**
  (diag_iter_max_auc par fold 711–1088). Vérifié par `usine/src/94_reviewer_arbres_E30.py` (log `usine/logs/94_rev_arbres_E30.log`, 5 folds) :
  Gini moyen 800 : 0,28195 | 1000 : 0,28155 | 1200 : 0,28095 | 1400 : 0,27984 | 1680 : 0,27820 ; 1680 vs 1400 : −0,0010 à −0,0025, 5/5 folds négatifs.
  Avec 25 % de données en plus, 1680 arbres sur 100 % correspond à peu près à 1400 sur 80 % : E80 est donc dans la **même** zone de sur-ajustement léger
  que E30 dans 2_306, pas pire. Ce n'est pas un bug. Correction possible (non validée, non nécessaire) : E80 à ~1000–1200 arbres.
  Ne pas l'appliquer dans l'urgence : ce serait un choix fait sur la CV, et l'effet sur le mélange est dilué (1 membre sur 4).

## 5. Autres points
- **IMPORTANT (décision lead-tech)** : 2_308 est le refit de **2_306**, pas de 2_307 (recommandé par la revue 93, +0,0003 CV). Le gain du refit n'est pas mesurable ;
  l'ordre de grandeur attendu du refit est comparable au +0,0003 de 2_307. Les deux sont déposables ; choisir 2_308 comme sélection finale est un pari non validé.
  Recommandation : déposer 2_308 pour obtenir son score serveur ; sélection finale = celle des deux qui est la meilleure au serveur, en gardant en tête que l'écart
  public < ~0,002 n'est pas significatif ; à défaut, 2_307 (validé en CV).
- MINEUR : `E83_refit_info.json` indique `moyenne_equivalent` 0,5 car `E32c_test.csv` est en rangs : sans effet.
- Non vérifié : ré-exécution complète des refits (temps) ; score serveur.
