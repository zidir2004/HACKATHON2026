# 05 — Analyste : analyse métier et explicabilité de la baseline E01

## Mission reçue
Du lead-tech : première analyse métier + explicabilité à partir de E01 (LightGBM, 57 variables brutes, Gini CV 0,2787) pour nourrir la soutenance (analyse approfondie /4, méthodologie /4).

## Ce qui a été fait
Script `usine/src/05_analyse.py` (config E01 réutilisée de `03_train.py`). Un modèle réentraîné sur les folds 1-4, validation = fold 0 (539 arbres, Gini 0,2766, cohérent avec E01). SHAP (TreeExplainer) sur 20 000 lignes du fold 0. Courbes ROC, gains et déciles sur l'OOF complet (595 212 lignes). Profils haut/bas risque. Taux par modalité des variables catégorielles.

## Résultats clés
1. **Les ps_calc pèsent peu** : 20 variables sur 57 (35 % des variables) mais seulement 8,1 % de l'importance SHAP (ind 43,6 % pour 18 variables, car 35,9 % pour 16, reg 12,5 % pour 3). Cohérent avec l'audit (calc = bruit). Attention : 8 % n'est pas 0 : le modèle utilise un peu ces variables, probablement pour ajuster du bruit. Le retrait est à confirmer par la CV, pas par SHAP seul.
2. **Variables dominantes (|SHAP| moyen)** : ps_car_13 (0,130), ps_ind_03 (0,089), ps_ind_05_cat (0,081), ps_ind_15 (0,079), ps_ind_17_bin (0,062), ps_reg_03 (0,058), ps_car_01_cat (0,055), ps_reg_01 (0,053), ps_ind_06_bin, ps_ind_04_cat. ps_car_13 est nettement en tête.
3. **ps_car_11_cat est sous-exploitée** : seulement 22e en SHAP alors qu'elle est citée en tête par l'audit (Gini 0,135 en encodage cible). Le modèle la lit comme un nombre (le code 1..104 n'a pas d'ordre) : argument concret pour l'encodage par taux cible ou le traitement catégoriel natif.
4. **Pouvoir de discrimination (OOF)** : Gini 0,2786 = AUC 0,639. Gains : les 10 % les plus risqués concentrent **21,7 %** des sinistres, 20 % -> 36,9 %, 30 % -> 48,7 %, 50 % -> 68,5 % (hasard : 10, 20, 30, 50 %).
5. **Déciles** : taux de sinistre de 1,55 % (décile 1) à 7,91 % (décile 10), soit un ratio **5,1**. Croissance régulière : le modèle est bien ordonné. Le taux moyen est 3,64 %.
6. **Profils** (10 % haut vs bas) : écarts les plus nets sur ps_car_13 (+1,9 écart-type), ps_reg_03, ps_ind_06_bin (plus bas chez les risqués), ps_ind_17_bin (38 % vs 0,9 %), ps_ind_05_cat, ps_reg_02, ps_car_15, ps_car_07_cat. Les manquants distinguent aussi les groupes : ps_ind_05_cat manquant 6,2 % chez les risqués vs 0,03 % ; ps_car_07_cat 13,6 % vs 0 % ; mais ps_reg_03 et ps_car_03_cat sont plus souvent manquants chez les peu risqués (50 % et 83 %). Confirme que « manquant » est une information. Limite : les moyennes de colonnes contenant -1 sont déformées par le -1 (voir `05_profils.csv`, colonnes de % manquant à lire en parallèle).
7. **Modalités catégorielles** : ps_car_11_cat a 104 modalités, effectifs de 851 à 63 780, taux de 2,0 % à 7,4 % (du simple au triple), sans ordre naturel. **Correction d'une idée reçue** : il n'y a pas de modalité vraiment rare ici (aucune sous 500 assurés, médiane de l'IC 95 % = ±0,67 pt), donc le lissage de l'encodage cible est moins critique pour cette variable que supposé ; il reste utile pour les modalités de 851-3 000 assurés (IC d'environ ±1 pt, comparable à l'écart entre modalités) et pour toute variable plus fine. À ne pas surjouer à l'oral.

## Décisions et justification
- SHAP sur le fold 0 et un échantillon de 20 000 lignes : suffisant pour des classements stables, coût faible. Valeurs en log-odds (échelle du modèle), pas en probabilité.
- Groupes par préfixe du nom (ind/reg/car/calc) : seule structure connue des variables anonymisées.
- Variables catégorielles à montrer : ps_car_11_cat (demandée), puis les 3 plus importantes en SHAP : ps_ind_05_cat, ps_car_01_cat, ps_ind_04_cat.

## Points d'attention
- Variables anonymisées : aucun sens métier certain. Hypothèses prudentes seulement : ps_ind_ = caractéristiques du conducteur/contrat, ps_car_ = véhicule (ps_car_13 pourrait être liée à la valeur ou la puissance du véhicule, simple supposition), ps_reg_ = zone géographique.
- SHAP = ce que le modèle utilise, pas une causalité.
- Aucun effet de période ou de collecte observable : pas de date dans les données ; l'audit ne trouve pas de dérive train/test et l'id n'est pas lié à la cible. Pas de variable suspecte nouvelle. Seul point : ps_ind_03 est 2e en SHAP alors qu'elle n'était pas citée dans l'audit comme tête de liste : variable ordinale, à surveiller mais rien d'anormal détecté.
- Le Gini 0,28 reste modeste (AUC 0,64) : le sinistre est un événement très aléatoire. À dire honnêtement au jury.
- Le Gini du fold 0 (0,2766) est un peu sous la moyenne CV (0,2787) : variation normale (écart-type 0,009).
- Analyse des faux positifs/négatifs sur seuil non faite (la métrique est le Gini, pas de seuil) ; remplacée par les déciles et profils.

## Recommandations
1. Retirer les ps_calc dans la prochaine expérience et la valider par CV appariée (argument SHAP : 8 % de l'importance pour 35 % des variables).
2. Traiter ps_car_11_cat en catégoriel natif LightGBM ou encodage cible fait dans les folds.
3. Conserver les indicateurs de manquant (info réelle, surtout ps_ind_05_cat, ps_car_07_cat).
4. Refaire cette analyse sur le meilleur modèle final pour comparer (ratio D10/D1, gains top 20 %).

## Fichiers produits
Code : `usine/src/05_analyse.py`. Log : `usine/logs/05_analyse_v2.log`.
Figures (`usine/figures/`) et message oral :
- `05_shap_importance.png` : « Voilà ce qui pèse dans la décision du modèle : ps_car_13 en tête, puis des variables conducteur. »
- `05_shap_beeswarm.png` : « Chaque point est un assuré ; rouge = valeur élevée ; on voit le sens de l'effet de chaque variable. »
- `05_shap_dep_ps_car_13.png`, `..._ps_ind_03.png`, `..._ps_ind_05_cat.png`, `..._ps_ind_15.png` : « Courbe de l'effet d'une variable sur le risque ; la ligne orange est l'effet moyen. » (formes à décrire en regardant la figure, non commentées ici)
- `05_importance_groupes.png` : « Les 20 variables calc, 35 % des variables, ne pèsent que 8 % : on peut les retirer. »
- `05_roc.png` : « AUC 0,64 / Gini 0,28 : mieux que le hasard, loin d'être parfait. »
- `05_gains.png` : « Les 20 % d'assurés les plus risqués selon le modèle concentrent 37 % des sinistres. »
- `05_deciles.png` : « Le taux de sinistre passe de 1,6 % à 7,9 % du décile 1 au 10 : le classement est cohérent. »
- `05_profils.png` : « Les assurés à risque se distinguent surtout par ps_car_13, ps_reg_03, ps_ind_17/06_bin. »
- `05_modalites_ps_car_11_cat.png` : « 104 modalités sans ordre, taux de 2 % à 7,4 % : à encoder par le taux plutôt que comme un nombre. »
- `05_modalites_autres_cat.png` : « Même constat pour les autres catégorielles ; les manquants (-1) ont leur propre taux. »
Données : `usine/rapports/05_resultats.json`, `05_shap_importance.csv`, `05_deciles.csv`, `05_profils.csv`, `05_modalites_*.csv`.
