# 10 — Analyste : analyse approfondie du sujet et plan de soutenance

## Mission reçue
Du lead-tech (10h40) : distributions (demande explicite du prof), analyse du modèle final E09, lecture métier prudente, plan de soutenance 12-15 min pour 5 membres, questions du jury. Cible : oral /13 (méthodologie /4, analyse approfondie /4, code /2, équipe /3). Aucun usage du GPU.

## Ce qui a été fait
Code : `usine/src/10_analyse_approfondie.py` (pandas, numpy, matplotlib ; s'exécute en moins de 2 min sans job). Figures `usine/figures/10_*.png` (8), tableaux `usine/rapports/10_*.csv` et `10_resultats.json`. Documents : `usine/soutenance/plan.md`, `usine/soutenance/questions_reponses.md`. Toutes les figures ont été relues visuellement ; corrigés : séparateurs décimaux en virgule, libellés tronqués, modalités manquantes rares hors échelle.

## Résultats clés
1. **Variables continues** (`10_continues.png`, taux par tranche, tranches de moins de 300 assurés masquées) : ps_car_13 : 2,3 % puis plus de 7 % (×3) ; ps_reg_02 : 2,4 % puis 6,2 % ; ps_reg_03 : 2,9 % puis 5,9 % ; ps_car_12 et ps_car_14 : hausse irrégulière 3-4 % puis 6-7 % ; **ps_ind_03 non monotone** (6,3 % pour la valeur 0, creux à 3 % pour 2-3, puis 4 %) ; ps_ind_15 en légère baisse (4,2 à 3,1 %) ; ps_reg_01 presque plat sauf le point 0. Manquants : ps_reg_03 (18 %) : 2,9 % de sinistres, ps_car_14 (7 %) : 4,0 %.
2. **Catégorielles et binaires** (`10_categorielles.png`, IC de Wilson à 95 %, effectifs en milliers) : ps_ind_17_bin = 1 : 5,45 % (54 000 assurés) contre 3,40 % ; ps_ind_06_bin = 1 : 2,88 % contre 4,15 % ; ps_car_07_cat manquant 7,9 % (8 600) contre 3,5 % ; ps_ind_05_cat manquant 8,1 % (4 300) contre 3,4 % ; ps_car_09_cat : 3,4 à 5,8 % selon la modalité. Les modalités rares (ps_car_04_cat = 7 : 100 assurés ; ps_car_01_cat manquant : 82 ; ps_ind_04_cat manquant : 66) ont des intervalles trop larges pour conclure : à dire à l'oral.
3. **Signal par groupe** (`10_signal_groupes.png`) : ind 18 variables / 44 % de l'importance SHAP ; reg 3 / 12 % ; car 16 / 36 % ; calc 20 / 8 %. Signal univarié médian : reg 0,107, car 0,076, ind 0,028, calc 0,003 (max 0,013).
4. **Modèle final E09** (OOF 446 409 assurés, Gini OOF 0,2833, AUC 0,6416) : top 10 % = 22,1 % des sinistres, **top 20 % = 37,2 %** (E01 : 36,9 %), top 30 % = 49,2 %, top 50 % = 68,4 %. Déciles : taux de 1,49 % (D1) à 8,04 % (D10), **ratio 5,4** (E01 : 5,1). Les 50 % les moins risqués (taux 2,3 %) ne portent que 31,6 % des sinistres. Le gain de E09 sur E01 est faible sur les gains cumulés (+0,3 point à 20 %) : cohérent avec +0,005 de Gini.
5. **Progression et ce qui a marché** (`10_progression_modeles.png`, `10_ce_qui_a_marche.png` ; Δ apparié, barre = ±2 erreurs-types, verdict recalculé avec la règle durcie du reviewer) : gains confirmés : E09 mélange (+0,0025, 5/5), E13 lr 0,01 (+0,0009, 5/5), E27 somme des binaires (+0,0012, 5/5 mais voir le point d'attention) ; échecs : E06 (−0,0099), E19 (−0,0028), E21 (−0,0013) ; tout le reste neutre (E02 : +0,0022 sur 4/5, intervalle ±0,0031).
6. **Chevauchement des scores** (`10_scores_par_classe.png`) : les assurés avec sinistre sont surtout dans les 20 % supérieurs, mais 32 % d'entre eux sont dans les 50 % inférieurs : c'est l'image de l'AUC de 0,64.

## Lecture métier (prudente)
- Hypothèses raisonnables, toutes non vérifiables : ps_ind = caractéristiques de l'assuré ou du contrat, ps_car = véhicule (ps_car_13 : peut-être valeur ou puissance, à ne jamais affirmer), ps_reg = zone géographique. ps_ind_06..09_bin sont des one-hot d'un même attribut (audit) : ps_ind_06_bin = 1 est protecteur (2,9 %).
- Manquants informatifs : ps_car_07_cat et ps_ind_05_cat manquants ×2,2 (profil atypique ou dossier incomplet) ; ps_reg_03 et ps_car_03_cat manquants ×0,75 (probablement liés à un canal ou une région). Hypothèse, pas une certitude.
- **Usage du score** : segmentation tarifaire, priorisation de contrôles ou d'actions de prévention ; un assuré du dernier décile a un taux 5,4 fois celui du premier. Ne pas l'utiliser seul pour refuser un assuré.
- **Limites** : AUC 0,64 (on classe une population, pas un individu) ; variables anonymisées ; pas de date donc ni effet de période ni test temporel (aucune dérive train/test, AUC adversaire 0,5008) ; 3,6 % de positifs ; CV optimiste de 0,0005-0,0008 (arrêt anticipé) et dépendante du découpage : annoncer 0,281-0,283.

## Décisions et justification
- Taux par tranche de même largeur, 20 tranches entre les centiles 0,5 et 99,5, tranches masquées sous 300 assurés : lisibilité sans sur-interpréter.
- Groupe « signal » = colonne `signal` de `audit_variables.csv` (max du Gini brut et du Gini en encodage cible dans les folds) ; part SHAP issue de E01 (pas de nouveau SHAP sur E09 : le mélange n'est pas directement explicable par TreeExplainer et le SHAP E01 couvre les mêmes variables).
- Progression : E00, E01, E02, E06, E09 comme demandé ; verdicts recalculés avec la règle durcie : ils concordent avec `comparaisons.csv` sauf E02 (neutre, déjà signalé).

## Points d'attention
- **Erreur dans mon rapport 05** : il parlait d'un OOF de 595 212 lignes ; le bon chiffre est 446 409 (595 212 = train + test). Les résultats (gains, déciles) étaient justes ; seul le libellé de la figure `05_gains.png` / du rapport est à corriger si cités.
- **E27 fragile** : +0,0012 avec une graine (5/5), mais E28 (3 graines) donne +0,0005 (3/5, neutre) et le mélange E24 avec F4 +0,0001 vs E09 (neutre). À ne pas présenter comme un gain acquis. Mon graphique ne reprend que la comparaison E27 vs E02 ; à relancer (`python -u usine/src/10_analyse_approfondie.py`, 1 min) si de nouvelles lignes arrivent dans `comparaisons.csv`.
- SHAP : calculé sur E01 (LightGBM) ; à dire à l'oral.
- Questions_reponses : deux points à confirmer par l'équipe (aucune soumission publique reprise ; rééquilibrage jamais testé).
- La répartition des 5 membres est une proposition : je ne connais ni les noms ni qui a fait quoi.
- Pas de variable suspecte nouvelle pour le reviewer.

## Recommandations
1. Figer E09 (ou E17 si le lead préfère, équivalents) avant 15h ; la soutenance ne doit pas dépendre d'un nouveau résultat.
2. Faire relire `plan.md` par l'équipe vers 14h pour ajuster la répartition et le minutage ; répétition chronométrée (14 min).
3. Si le temps le permet : un SHAP sur E09 n'est pas nécessaire ; mieux vaut une slide sur le score serveur final.

## Figures et « ce qu'il faut dire à l'oral »
- `10_continues.png` : « Pour chaque variable, les barres montrent comment les assurés se répartissent et la courbe orange le taux de sinistre dans chaque tranche. Pour ps_car_13, le risque triple quand la valeur monte ; pour ps_ind_03, la relation n'est pas régulière. »
- `10_categorielles.png` : « Chaque point est le taux de sinistre d'une modalité, avec son intervalle de confiance : plus le groupe est petit, plus l'incertitude est grande. Les manquants (losanges rouges) ont leur propre taux, souvent plus élevé. »
- `10_signal_groupes.png` : « Les familles ind, reg et car portent le signal ; les 20 variables calc sont presque sans information, et le modèle leur donne 8 % de son attention. »
- `10_progression_modeles.png` : « Le gros progrès vient d'avoir choisi des arbres plutôt qu'une régression ; ensuite, on joue au millième, et les barres d'erreur montrent pourquoi on compare pli par pli. »
- `10_ce_qui_a_marche.png` : « Chaque point est un essai : à droite de zéro, mieux que sa référence ; la barre est l'incertitude. Seuls trois essais sont clairement verts ; plusieurs sont rouges ; la majorité ne change rien. »
- `10_gains_E09.png` : « Si on classe les assurés par score, les 20 % du haut contiennent 37 % des sinistres, contre 20 % au hasard. »
- `10_deciles_E09.png` : « Du décile 1 au décile 10, le taux de sinistre passe de 1,5 % à 8 % : le score est bien ordonné. »
- `10_scores_par_classe.png` : « Les deux populations se recouvrent beaucoup : on classe, on ne tranche pas individuellement. »

## Fichiers produits
Code : `usine/src/10_analyse_approfondie.py`. Figures : `usine/figures/10_continues.png`, `10_categorielles.png`, `10_signal_groupes.png`, `10_progression_modeles.png`, `10_ce_qui_a_marche.png`, `10_gains_E09.png`, `10_deciles_E09.png`, `10_scores_par_classe.png`. Données : `usine/rapports/10_continues_tranches.csv`, `10_categorielles_modalites.csv`, `10_deciles_E09.csv`, `10_segments_E09.csv`, `10_ce_qui_a_marche.csv`, `10_resultats.json`. Soutenance : `usine/soutenance/plan.md`, `usine/soutenance/questions_reponses.md`.
