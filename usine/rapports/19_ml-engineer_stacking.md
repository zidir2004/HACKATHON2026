# 19 — ML engineer — stacking par méta-modèle (E90–E93)

**Hypothèse (idée du professeur)** : pondérer les 4 membres de 2_306 (E30 LGB, E07 XGB, E08 CatBoost, E32c réseau DAE) selon le profil de l'assuré bat la moyenne fixe des rangs (E33f).

**Protocole** : entrées = rangs percentiles des prédictions (OOF et test rangés séparément) + 6 variables de profil (nb_manquants, ps_car_13, ps_ind_03, ps_reg_03, ps_ind_05_cat==0, ps_ind_17_bin ; imputation médiane et standardisation apprises sur les folds d'entraînement). Évaluation emboîtée : pour chaque fold k, méta-modèle ajusté sur l'OOF des 4 autres folds, prédit le fold k. Les 3 méta-modèles ont été fixés avant tout résultat. Comparaison appariée vs E33f (garde si 5/5 ou Δ > 2·sd/√5).

| Exp | Méta-modèle | Gini CV (moy ± std) | Δ vs E33f | Folds gagnés | Verdict |
|---|---|---|---|---|---|
| E90 | Logistique sur 4 rangs | 0,28531 ± 0,00738 | -0,00000 | 1/5 | neutre |
| E91 | Logistique rangs + profil + interactions rang×profil | 0,28287 ± 0,00800 | -0,00243 | 0/5 | rejete |
| E92 | LightGBM 6 feuilles, min 2000, lr 0,02, 300 arbres | 0,28289 ± 0,00722 | -0,00242 | 0/5 | rejete |

**Soumission** : E93 = E90 réentraîné sur tout l'OOF -> `usine/submissions/2_310.csv` ; remise à l'échelle par quantile mapping (valeurs triées de la moyenne des probas E30/E07/E08/E32b ; E32c_test est en rangs). Format contrôlé (148 803 lignes, sans en-tête, ordre test.csv, LF, 0 NaN, [0,0075 ; 0,3946], moy 0,0367). Spearman vs 2_306 = 1,00000 : c'est pratiquement la même soumission.

**Interprétation (oral)** : la logistique apprend des poids quasi égaux (E30 0,48 ; E07 0,42 ; E08 0,42 ; réseau 0,44) — la moyenne à poids égaux était déjà optimale. Les interactions profil existent mais sont faibles (ex. le réseau E32c pèse un peu moins quand ps_ind_05_cat = 0 (-0,09) et un peu plus quand ps_reg_03 est élevé (+0,05) ; nb_manquants ne change presque rien : poids XGB 0,39 -> 0,45 de -1 à +1 écart-type) et elles n'améliorent pas la CV : elles surajustent du bruit. Coefficients complets : `usine/experiences/E91_coefficients.csv`. Conclusion : les 4 modèles se trompent sur les mêmes profils ; aucun segment où un modèle domine durablement (cohérent avec E95–E99, sélection par segment, rejetée).

**Temps** : 13 s CPU. **Prochaine idée** : aucune en stacking ; garder 2_306/2_307 comme choix CV.
