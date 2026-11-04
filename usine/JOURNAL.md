# Journal — Hackathon M2 DS4SC : prédiction de sinistres (Gini)

## 0. Résumé (toujours à jour)
- **Modèle final présenté : E33f → `2_306.csv`, score serveur 0,30079** (meilleur score officiel) — Gini CV 0,2853 ± 0,0074
- Modèle : rangs égaux de E30 (LightGBM Jahrer + one-hot), E07 (XGBoost GPU), E08 (CatBoost GPU), E32c (réseau sur autoencodeur débruiteur, 3 graines)
- 2_307 (E56) : serveur 0,3003, CV 0,2856 → égalité dans le bruit avec 2_306, on garde le plus simple
- Livrables : rapport d'équipe (Claude Docs « Rapport Hackathon M2 — Soumission 2_306 »), `soutenance/plan.md`, `questions_reponses.md`, figures `11_*`, `src/README_2_306.md`, code commenté
- En cours : 3ᵉ autoencodeur E70 (GPU), pour information seulement
- Échéance : **16h00 (Paris) fermeture des soumissions** ; gel à 15h ; check-ups 11h et 14h

## 1. Compréhension du sujet
Voir `00_brief.md`. Classification binaire en probabilité, Gini = 2·AUC−1, 3,64 % de positifs, soumission CSV sans en-tête, 148 803 lignes triées par id.

## 2. Audit des données
Rapport : `rapports/01_data-engineer_audit.md`, tableaux `rapports/audit_*.csv`, figures `figures/01_*.png`.
- Train 446 409 × 59, test 148 803 × 58, 16 271 positifs (3,645 %). Aucun doublon, aucun id commun train/test.
- 13 variables avec des -1 (manquants), surtout ps_car_03_cat 69 %, ps_car_05_cat 45 %, ps_reg_03 18 %. **Le manque est informatif** : sinistres ×0,75 quand ces 3 sont manquantes, ×2,2 pour ps_car_07_cat / ps_ind_05_cat.
- **ps_calc_* = bruit** : |Gini| univarié max 0,007 ; LightGBM calc seules = Gini 0,002.
- Meilleures variables : ps_car_13 (Gini 0,150), ps_car_11_cat (0,135 si encodé par taux cible dans les folds vs 0,026 brut), ps_car_12, ps_reg_03, ps_reg_02.
- **Pas de dérive** train/test (adversarial AUC 0,5008, PSI max 0,001) → la CV est un bon estimateur du score test.
- Structures : ps_ind_14 = somme de ps_ind_10..13_bin ; ps_ind_06..09_bin et ps_ind_16..18_bin sont des one-hot ; ps_car_15, ps_car_12, ps_car_14, ps_reg_03 sont des racines carrées d'entiers.


## 3. Décisions sur les données
| # | Problème constaté | Décision | Justification | Validé par | Impact mesuré |
|---|---|---|---|---|---|
| 1 | 20 ps_calc_* sans signal | **Retirées (v1)** | Bruit (Gini seules 0,002 ; 8 % du SHAP pour 35 % des variables) ; modèle plus simple | Utilisateur 09h32 | E02 +0,0022 (4/5 folds) — non significatif, gardé comme simplification |
| 2 | -1 = manquants informatifs | Gardés tels quels + indicateurs testés (v2) | Ne pas effacer de signal | Utilisateur | E03 −0,0009 → indicateurs **non retenus** (les arbres voient déjà le -1) |
| 3 | Redondances (ps_ind_14, one-hot) | Testé (v3) | Lisibilité | Utilisateur | E04 neutre → non retenu |
| 4 | ps_car_11_cat (104 modalités) lue comme nombre | Encodage cible dans les folds (E05), catégorielles natives (E06) | Gini univarié 0,026 → 0,135 si encodée | Utilisateur | E05 neutre (+0,0003) ; E06 rejeté (−0,010) |
| 5 | Imputation, écrêtage, suppression de lignes, rééquilibrage | **Non faits** | Détruiraient du signal ; inutiles pour le Gini et les arbres | Utilisateur | — |

## 4. Protocole de validation
- StratifiedKFold 5 folds, shuffle, graine 42, figé dans `experiences/folds.csv` : toutes les expériences utilisent les mêmes folds.
- Métrique : Gini = 2·AUC−1 par fold, on rapporte moyenne ± écart-type + Gini OOF. Pas de seuil (seul l'ordre compte).
- Stratifié car seulement 3,6 % de positifs (chaque fold garde la même proportion). Transformations apprises (encodage cible…) faites dans les folds uniquement.
- Prédiction test = moyenne des 5 modèles de fold.
- Règle de décision : un gain < écart-type entre folds (~0,008) n'est pas un gain ; pour les petits gains on regarde aussi la différence **fold par fold** avec la référence (demande du reviewer).
- Réserve du reviewer : early stopping sur le fold de validation → CV optimiste de ~0,0005–0,0008 ; selon la graine du découpage, E01 vaut 0,2762–0,2787. Annoncer ~0,277–0,279.


## 5. Expériences
| Exp | Heure | Changement testé | Gini CV (± std) | Δ vs meilleur | Verdict |
|---|---|---|---|---|---|
| E00 | 09h18 | Régression logistique (repère naïf) | 0,2392 ± 0,0066 | — | repère |
| E01 | 09h18 | LightGBM, données brutes v0 (57 variables, -1 tel quel) | **0,2787 ± 0,0089** (OOF 0,2786) | +0,039 vs E00 | garde |
| E02 | 09h50 | v1 : sans ps_calc_* | 0,2809 ± 0,0077 | +0,0022 (4/5) | garde (simplification) |
| E03 | 09h50 | v2 : + indicateurs de manquants | 0,2800 ± 0,0082 | −0,0009 | neutre |
| E04 | 09h50 | v3 : redondances nettoyées | ≈0,2806 | −0,0003 | neutre |
| E05 | 09h55 | + encodage cible ps_car_11_cat (dans les folds) | 0,2812 | +0,0003 (2/5) | neutre |
| E06 | 09h55 | _cat en catégorielles natives LightGBM | 0,2710 ± 0,0077 | −0,0099 (0/5) | rejeté |
| E07 | 10h00 | XGBoost GPU sur v1 | 0,2807 | ≈0 | neutre seul |
| E08 | 10h00 | CatBoost GPU sur v1 | ≈0,281 | ≈0 | neutre seul |
| **E09** | 10h00 | **Moyenne des rangs E02+E07+E08 (poids égaux)** | **0,2834 ± 0,0076** (OOF 0,2833) | **+0,0026 (5/5)** | **garde** |
| E10–E12 | 10h00 | Réglages LightGBM (feature_fraction, num_leaves, min_child_samples) | 0,2804–0,2810 | ≤0 | rejetés/neutres |
| E13 | 10h00 | LightGBM lr 0,02 → 0,01 | 0,2818 ± 0,0077 | +0,0003 vs E05 | garde (seul) |
| E14 | 10h00 | Mélange E13+E07+E08 | 0,2837 | +0,0003 (3/5) vs E09 | neutre |
| E15–E16 | 10h00 | CatBoost / XGBoost lr plus faible | ≈0,2811 | − | neutres |
| E17a | 10h30 | LightGBM E13 moyenné sur 3 graines | 0,2822 | +0,0004 (3/5) vs E13 | neutre |
| E17 | 10h30 | Mélange E17a + E07 + E08 | 0,2837 ± 0,0077 | +0,0003 (4/5) vs E09 | neutre |
| E18 | 10h35 | Poids non égaux (choisis sur 4 folds, testés sur le 5e) | — | +0,0001 | abandonné |

**Lecture E17/E18** : changer la graine d'un modèle ne fait varier le Gini que de 0,0002, contre 0,0077 d'écart entre folds → la variance vient des données, pas du modèle. Des poids optimisés n'apportent rien face à des poids égaux. Le mélange E09 est donc stable.
| E19 | 10h45 | v4 complète (12 variables d'interaction/comptage) | ≈0,2781 | −0,0028 (0/5) vs E02 | rejeté |
| E20–E23, E25–E26 | 10h50 | Familles v4 une par une (croisement F5, ratios car13 F2, comptages F1, F2b, entiers F3) | ≈0,280–0,281 | −0,0013 à +0,0003 | neutres / F2 rejetée |
| E27 | 10h52 | + ps_ind_bin_sum (F4) | 0,2821 | +0,0012 (5/5) graine 42 | **faux positif** : +0,0005 (3/5) en moyenne sur 3 graines (E28) → neutre |
| E24 | 10h58 | Mélange XGB+CatBoost+LGBM sur v1+F4 | 0,2835 | +0,0001 (3/5) vs E09 | neutre |

**Lecture v4** : aucune variable construite n'aide ; les arbres trouvent déjà ces interactions. E27 montre qu'un gain « 5/5 folds » peut venir de la graine : pour les variables, on exige désormais la confirmation sur 3 graines (`src/09_verif_graines.py`). Les 3 boostings sont corrélés à 0,96–0,99 : le mélange est proche de son plafond.

**Règle de verdict durcie (demande reviewer, 10h35)** : `garde` seulement si 5/5 folds gagnés ou Δ moyen > 2·sd(Δ)/√5. Avec cette règle, E02 (retrait calc) devient `neutre` → présenté comme simplification sans perte.

**Lecture** : les retouches de données n'apportent presque rien aux arbres (ils gèrent déjà -1 et redondances). Le seul gain net vient du **mélange de 3 algorithmes différents** : leurs erreurs ne sont pas les mêmes, la moyenne lisse le bruit (gain sur les 5 folds, et le mélange bat le meilleur des 3 modèles dans chaque fold).

### Pistes inspirées de la solution 1re place Kaggle (M. Jahrer) — 11h05–12h55
Source fournie par l'utilisateur : blend 1 LightGBM + 5 réseaux sur autoencodeur débruiteur (DAE), calc retirées, one-hot des _cat, moyenne simple.
| Exp | Changement | Gini CV | Δ apparié | Verdict |
|---|---|---|---|---|
| E29 | LightGBM paramètres Jahrer, 1400 arbres fixes | 0,2788 | −0,0030 vs E13 (0/5) | rejeté (1400 arbres = trop pour 357k lignes) |
| E30b / E32A | + one-hot des _cat (colonnes d'origine gardées), confirmé 3 graines | 0,2805 | +0,0011 (4/5) | garde |
| E35/E36 | 1050 arbres (mise à l'échelle) | 0,2821 | +0,0016 (5/5) | garde, = E13 |
| E34 | Mélange E32A + E07 + E08 | 0,2843 | +0,0008 vs E09 (5/5) | garde → 2_305 |
| E32 | Réseau 4500-1000-1000 sur activations DAE (early stopping) | 0,2711 | — | seul plus faible, mais très différent (corr. rangs 0,87) |
| E32b/E32c | 6 époques fixes, 3 graines | 0,2752 | +0,0030 vs E32 | membre retenu |
| **E33f** | **Mélange E30 + E07 + E08 + E32c** | **0,2853 ± 0,0074** | **+0,0019 vs E09, +0,0011 vs E34 (5/5)** | **garde → 2_306** |
| E40/E41 | DAE ré-entraîné 38 min de plus (1320 époques) | 0,2851 | −0,0002 vs E33f (1/5) | rejeté (DAE déjà convergé) |

| E50/E51c | 2ᵉ DAE (swap noise 7 %, 28 min GPU) + réseau, 3 graines | 0,2770 (seul) | +0,0018 vs E32c (4/5) | garde (meilleur réseau) |
| E52 | MLP direct sans DAE, 3 graines | 0,2649 | fait baisser les mélanges (−0,0008 à −0,0015) | rejeté |
| **E56** | **E30 + E07 + E08 + moyenne(E32c, E51c)** (réseaux = 1/4 du poids) | **0,2856 ± 0,0073** | **+0,0003 vs E33f (4/5)**, bootstrap 90 % [+0,00005 ; +0,0005] | **garde → 2_307** |
| E55/E59 | E51c en 5ᵉ membre / à la place de E32c | ≈0,2855 | neutre | non retenus |
| E60–E67 | LightGBM DART, extra_trees, logistique riche, ExtraTrees sklearn ajoutés au mélange | 0,253–0,279 seuls | −0,0001 à −0,0023 | rejetés (trop faibles ou redondants) |

**Lecture** : un modèle *plus faible seul* (réseau 0,275) mais *différent* (il apprend une autre représentation des données) améliore le mélange : c'est la diversité qui paie, pas la force individuelle. Incident : collision d'identifiants E32/E33 entre les deux pistes parallèles → lignes de la piste A renommées E32A/E33A ; le reviewer a vérifié qu'aucun livrable n'utilise un fichier écrasé.

## 6. Revues du reviewer
- 10h14 — Revue E02–E16 / 2_303.csv (`rapports/91_reviewer_revue_E09.md`) : **déposable**. Pas de fuite (E05 bien en folds avec K-fold interne ; mêmes folds pour les 3 membres). Gain du mélange réel sur la graine 42 (5/5). Réserves : robustesse sur une autre graine en cours (job `91_revue_E09_s7`) ; gain E02 non significatif (à présenter comme simplification).
- 10h03 — Lead-tech : le mélange sortait des rangs (moyenne 0,5) ; ajout de `src/07_rangs_vers_proba.py` qui conserve l'ordre exact (même Gini) et donne des valeurs de probabilité (moyenne 0,0364), pour respecter « prédiction en probabilité ».
- 09h29 — Revue E01 / 2_302.csv (`rapports/90_reviewer_revue_E01.md`) : **déposable**. Format OK (vérifié par script indépendant), pas de fuite, résultats reproduits à l'identique. Mineur : fins de ligne CRLF (à passer en LF). Important : CV légèrement optimiste (early stopping + graine favorable) ; juger les petits gains fold par fold.

- 12h45 — Revue pistes A/B (`rapports/92_reviewer_revue_pistesAB.md`) : **2_305 et 2_306 déposables, 2_306 à privilégier**. Gain E33f réel (bootstrap 90 % : +0,0004 à +0,0016 vs E34). DAE sur train+test sans cible : pas de fuite, pas d'optimisme. Annoncer ~0,284–0,285 (correction de sélection).

- 14h10 — **Revue finale** (`rapports/93_reviewer_revue_finale.md`) : 2_307 déposable, **à déposer en dernier** ; 2_306 repli. Gain E56 petit mais stable sur les 3 graines du réseau. 2_306 et 2_307 corrélés à 0,9995 sur le test. Chaîne reproductible (commandes §5 du rapport). Ne pas déposer `_check_2_305_regen.csv`.

## 7. Livrables / soumissions
| Heure | Expérience | Gini CV | Score serveur | Fichier |
|---|---|---|---|---|
| 09h29 | E01 | 0,2787 | à venir | `submissions/2_302.csv` |
| 10h14 | E09 | 0,2834 | à venir | `submissions/2_303.csv` |
| 11h40 | E34 | 0,2843 | — | `submissions/2_305.csv` |
| 12h45 | E33f | 0,2853 | **0,30079** | `submissions/2_306.csv` |
| 14h10 | E56 | 0,2856 | 0,3003 | `submissions/2_307.csv` |

- Feature engineering v4 (10h20, `rapports/07_data-cleaner_features_v4.md`) : v1 + 12 variables non supervisées (comptages de catégorielles, interactions ps_reg_01×ps_reg_02, ps_car_13×ps_reg_03, ps_car_13/ps_car_12, ps_ind_03 & ps_ind_15, entiers reconstruits, somme des binaires, comptage du croisement ps_ind_05_cat×ps_car_04_cat). Plus prometteuse : `ind05_x_car04_count` (Gini univarié 0,124 vs 0,043/0,075 pour ses sources). À tester (E19+).

Historique serveur fourni par l'utilisateur (heures de dépôt) : 11h10 0,29273 · 12h34 0,29689 · 12h41 0,29681 · 12h58 **0,30079 (2_306)** · 14h06 0,29823 · 14h10 0,3003 (2_307). Correspondance des autres heures à confirmer.

## 8. Enseignements métier
Rapport `rapports/05_analyste_E01.md`, figures `figures/05_*.png`.
- Gini 0,28 (AUC 0,64) : le modèle **classe** bien sans être très précis individuellement. Les 20 % d'assurés les plus risqués concentrent 37 % des sinistres ; taux de sinistre de 1,6 % (décile 1) à 7,9 % (décile 10), ratio 5,1.
- Variables dominantes (SHAP) : ps_car_13, ps_ind_03, ps_ind_05_cat, ps_ind_15. Les ps_calc_* = 8,1 % de l'importance pour 35 % des variables.
- ps_car_11_cat : taux de sinistre de 2 % à 7,4 % selon la modalité sans ordre, mais les arbres s'en sortent déjà (E05 neutre).

## 9. Arguments pour la soutenance

## 10. Chronologie
- 09h05 — Démarrage de l'usine. Sujet lu (8 slides). Brief rédigé.
- 09h10 — Point de contrôle n°1 validé : groupe 2, soumission 2_302 ; on utilise le savoir Kaggle comme pistes mais on vérifie tout en CV ; calcul sur le PC de l'utilisateur via un lanceur de jobs (`usine/runner.py`).
- 09h10 — PC : 32 CPU, 32 Go RAM, RTX 4060. Bibliothèques installées. Lanceur `runner.py` actif.
- 09h12 — Audit (data engineer) et baseline (ML engineer) lancés en parallèle. Protocole figé.
- 09h24 — Audit terminé. E00 = 0,239, E01 = 0,2787.
- 09h29 — Revue : 2_302.csv déposable. Point de contrôle n°2 (décisions sur les données) présenté à l'utilisateur.
- 09h32 — Point de contrôle n°2 : 4 transformations validées (retrait calc, indicateurs manquants, catégorielles, redondances) ; pas d'imputation / écrêtage / suppression / rééquilibrage.
- 09h33 — Data cleaner (v1–v3) et analyste (SHAP E01) lancés en parallèle ; terminés 09h38.
- 09h40 — ML engineer : boucle E02→E16. Fin 10h00 : E09 (mélange 3 modèles) = 0,2834.
- 10h03 — Conversion rangs → probabilités (07_rangs_vers_proba.py), 2_303.csv.
- 10h14 — Revue : 2_303.csv déposable.
- 10h15 — Lancés en parallèle : data cleaner (v4) et ML engineer (multi-graines E17).
- 10h16 — Incident : 3 jobs GPU (2 du reviewer pour la robustesse graine 7/42, 1 du ML engineer) bloqués mutuellement sur la RTX 4060. Fold 0–1 de la graine 7 avaient fini (mélange cohérent : lgb 0,2768 / xgb 0,2753 / cb 0,2786 au fold 1).
- 10h38 — Lanceur v2 (`runner.py`) : un seul job GPU à la fois (préfixe `#gpu`), annulation par fichier `.kill`. Redémarrage demandé à l'utilisateur.
- 10h38 — `2_304.csv` (E17) prête mais quasi identique à 2_303 (corrélation de rang 0,9998) : pas d'intérêt à la déposer.
- 10h40 — Lanceur v2 actif. ML engineer (v4, E19–E28) et analyste (distributions, plan de soutenance) lancés en parallèle.
- 10h58 — v4 : rien ne passe (E27 faux positif démasqué par le contrôle multi-graines). Analyste : figures `10_*`, `soutenance/plan.md`, `soutenance/questions_reponses.md`.
- 11h02 — Sauvegarde GitHub (https://github.com/zidir2004/HACKATHON2026, sans les données). PyTorch GPU installé.
- 11h05 — Piste Kaggle (Jahrer) fournie par l'utilisateur : piste A (LightGBM + one-hot, CPU) et piste B (DAE + réseau, GPU) en parallèle.
- 12h05 — E33f = 0,2853. Collision d'IDs corrigée (E32A/E33A).
- 12h45 — Revue : 2_306 déposable. 12h55 — reprise DAE rejetée (E41).
- 12h57 — L'utilisateur donne carte blanche pour maximiser le score jusqu'au gel.
- 12h58 — Carte blanche : piste GPU (E50–E59) et piste CPU (E60–E69) en parallèle, plages d'IDs réservées pour éviter une nouvelle collision.
- 13h48 — E56 (2 réseaux DAE) = 0,2856. Modèles CPU additionnels tous rejetés.
- 14h04 — Vérification à la demande de l'utilisateur : la RTX 4060 est bien utilisée (torch cuda, 3,1 s/époque DAE) ; au repos depuis 13h48.
- 14h10 — Revue finale : 2_307 recommandé. Point de contrôle n°3 présenté.
- 14h12 — Scores serveur : 2_306 = 0,30079 > 2_307 = 0,3003. Décision utilisateur : **2_306 = modèle final présenté**. 3ᵉ DAE (E70) laissé tourner sur GPU.
- 14h20 — Analyste : figures `11_*` (progression, schéma, corrélations, gains/déciles E33f, ce qui a marché, apport des membres), plan de soutenance 5 membres, 22 questions-réponses. ML engineer : 7 scripts de la chaîne 2_306 commentés (AST inchangé vérifié), `src/README_2_306.md`.
- 14h25 — Rapport d'équipe rédigé dans Claude Docs (process, répartition des 5 membres, code, résultats).
