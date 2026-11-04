# 17 — Refit des 4 membres de 2_306 sur 100 % du train (E80–E84) -> 2_308

**Hypothèse** : réentraîner chaque membre de E33f sur 100 % du train (446k lignes au lieu de 357k par fold)
améliore légèrement le score test. **Changement** : un seul modèle par membre, entraîné sur tout le train, à
la place de la moyenne des 5 modèles de fold. Préparation et paramètres strictement identiques (code réutilisé de
03_train.py / 13_nn_dae.py). Pas de CV possible : contrôle par Spearman vs la prédiction test par folds.

| Exp | Membre | Choix du nombre d'itérations | Durée | Spearman vs équivalent folds | Moyenne proba |
|---|---|---|---|---|---|
| E80 | LightGBM E30 | 1680 = 1400 x 1,2 (taille du train x 1,25 ; choix a priori, non validé) | 15 s CPU | 0,99457 (E30) | 0,03645 |
| E81 | XGBoost E07 | 465 = moy. best_iteration (462, 382, 321, 256, 517 -> 387,6) x 1,2 | 4 s GPU | 0,99190 (E07) | 0,03646 |
| E82 | CatBoost E08 | 1068 = moy. (687, 778, 995, 826, 1165 -> 890,2) x 1,2 | 79 s GPU | 0,99102 (E08) | 0,03616 |
| E83 | Réseau DAE E32c | 6 époques fixes moyennées, 3 graines ; DAE dae.pt gelé, non réentraîné | 84 s GPU | 0,99779 (E32c) | 0,03789 |
| E84 | Rangs égaux E80-E83 + quantile mapping | — | s | 0,99815 vs 2_306 | 0,03674 |

Aucun repli utilisé : les 4 membres sont réentraînés. Tous les Spearman > 0,97 attendu (aucun bug détecté).
Le réseau garde 6 époques alors que chaque époque contient 1,25x plus de pas (lr 1e-4, batch 256 inchangés).

**Soumission** `usine/submissions/2_308.csv` : 148 803 lignes, sans en-tête, ids dans l'ordre de test.csv, LF,
0 NaN, [0,0071 ; 0,4365], moyenne 0,03674 (taux train 0,03645).

**Verdict** : `info` — gain non mesurable en CV. Le refit est une pratique standard (plus de données, mêmes
réglages) mais le facteur 1,2 sur les arbres est un a priori. Attention : la revue 93 recommande 2_307 (E56,
qui inclut le réseau E51c sur le 2e DAE) ; 2_308 est le refit de 2_306, PAS de 2_307. Pour un refit de 2_307, il
faudrait réentraîner aussi le réseau E51c (dae_swap07.pt) puis combiner via 18_combiner.py — non fait (hors mission).

**Outils** : `usine/src/17_refit_complet.py --exp E80|E81|E82|E83` ; `usine/src/18_combiner.py --membres <csv test> --sortie <csv> [--reference ...] [--sauver-melange ...]`
(rangs égaux, quantile mapping, contrôles format/LF/Spearman). Note : 18_combiner fait une moyenne à plat de ses
membres ; pour reproduire la structure E56 (moyenne de 2 réseaux = un seul membre) il faut d'abord moyenner ces réseaux.

**Prochaine idée** : si le temps le permet, refit du réseau E51c et combinaison E80+E81+E82+moyenne(E83,E51c refit).
