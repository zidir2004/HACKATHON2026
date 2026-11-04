# Data engineer — audit (2026-11-04 09:35)

## Mission reçue
Auditer `train.csv`, `test.csv` et `submission_example.csv` **sans rien modifier ni corriger** : typage et copie parquet, profil des variables, signal univarié, dérive train/test, doublons, corrélations, codes bizarres, colonnes déduites, structure des variables continues, figures, et liste de corrections possibles à décider avec l'utilisateur.

## Ce qui a été fait
- Script `usine/src/01_audit.py` (numéroté en 7 sections, tourne en 2 min sur le PC, code retour 0) + deux petites sondes `01_audit_extra.py` et `01_audit_extra2.py`.
- Les fichiers d'origine n'ont pas été touchés. Les copies typées sont dans `usine/data/interim/` (non corrigées : les -1 sont toujours là). Contrôle : le typage ne change aucune valeur (écart max 2e-7 sur les float32, 0 sur les entiers) ; mémoire 201 Mo -> 39 Mo.
- Pour le signal des variables « catégorielles », l'encodage par le taux de cible est calculé **uniquement sur les 4 autres folds** (CV 5 folds, seed 42, mêmes folds que `folds.csv` de l'ml-engineer). J'ai aussi mesuré le **bruit** : cible mélangée 300 fois, pour savoir à partir de quel Gini une variable se distingue du hasard.

## Résultats clés (chiffres)
**Cohérence et cible**
- train 446 409 × 59, test 148 803 × 58, mêmes colonnes dans le même ordre, 0 valeur vide (NaN) : les manquants sont uniquement codés **-1**. Aucun id en double, **0 id commun** train/test, test trié par id croissant, `submission_example.csv` : 148 803 lignes, ids identiques au test, sans en-tête (valeur constante 0,03).
- Cible : 16 271 positifs sur 446 409 = **3,645 %** (déséquilibre 1 pour 26). Aucun lien avec l'id : Gini de l'id = 0,0007, taux par décile d'id entre 3,44 % et 3,72 % (bruit).
- 57 variables : 47 entières (int8), 10 en float32. Aucune variable constante.

**Manquants (-1)** : 13 variables sur 57 en ont (0 chez les `ps_calc_*`, 0 chez `ps_ind` binaires).
| Variable | % manquant (train / test) | taux de sinistre manquant vs renseigné |
|---|---|---|
| ps_car_03_cat | 69,1 / 69,0 | 3,28 % vs 4,46 % (x0,74) |
| ps_car_05_cat | 44,8 / 44,7 | 3,18 % vs 4,02 % (x0,79) |
| ps_reg_03 | 18,1 / 18,0 | 2,87 % vs 3,82 % (x0,75) |
| ps_car_14 | 7,2 / 7,1 | 4,03 % vs 3,61 % (x1,12) |
| ps_car_07_cat | 1,9 / 1,9 | 7,87 % vs 3,56 % (x2,2) |
| ps_ind_05_cat | 1,0 / 1,0 | 8,14 % vs 3,60 % (x2,3) |
- Variables très rarement manquantes mais à fort lift : ps_ind_04_cat (66 lignes, x10), ps_car_01_cat (82 lignes, x8), ps_ind_02_cat (167 lignes, x5), ps_car_09_cat (416 lignes, x2,6). Effectifs faibles = intervalles de confiance très larges, ne pas sur-interpréter. ps_car_02_cat (4), ps_car_11 (3), ps_car_12 (1) : négligeables.
- **Être manquant est informatif** : l'indicateur « manquant » de ps_car_03_cat a un Gini de 0,07, du même ordre que la variable elle-même (0,077). Le nombre de -1 par ligne a un Gini de -0,072 (4,5 % de sinistres avec 0 manquant, 2,8 % avec 3).
- Blocs de manquants simultanés : (ps_ind_04_cat, ps_car_01_cat) r=0,86 ; (ps_ind_05_cat, ps_car_07_cat) r=0,67 ; (ps_car_03_cat, ps_car_05_cat) r=0,60 ; (ps_ind_02_cat, ps_ind_04_cat) r=0,60. Les manquants de train et de test ont les mêmes taux (écart max 0,14 point).

**Signal univarié** (Gini = 2·AUC-1 ; bruit : écart-type 0,0036 ; seuil à 5 % sur les 57 variables = 0,014)
- Top 10 (meilleur de « Gini brut » et « Gini encodage cible CV ») : ps_car_13 (0,150), ps_car_11_cat (**0,135** après encodage cible, mais 0,026 brut : la variable est non monotone), ps_car_12 (0,118), ps_reg_03 (0,110), ps_reg_02 (0,107), ps_car_01_cat (0,100), ps_car_14 (0,097 ; 0,017 brut), ps_car_15 (0,094), ps_ind_03 (0,089 ; 0,026 brut), ps_ind_06_bin (0,086). Détail des 57 : `audit_variables.csv`.
- Par groupe (signal moyen) : reg 0,096 > car 0,077 > ind 0,038 > **calc 0,004**.
- **Les ps_calc_* sont du bruit dans nos données** : aucune des 20 ne dépasse le seuil (|Gini| brut max 0,0069 ; encodage cible max 0,0128, inférieur au bruit 0,018) ; corrélation max d'une calc avec n'importe quelle autre variable = 0,005. Test global avec LightGBM (5 folds, mêmes paramètres) : toutes variables **0,2756 ± 0,0075**, sans calc **0,2773 ± 0,0074**, calc seules **0,0021 ± 0,0042** (= 0, aléatoire).
- **Pas de fuite** : aucune variable avec Gini > 0,15 (donc loin de AUC 0,9), id non corrélé à la cible, aucune variable « postérieure à l'événement » identifiable (variables anonymisées, donc à ne pas affirmer au-delà du test statistique).

**Dérive train/test** : pas de dérive. PSI max 0,0011 (seuil d'alerte 0,1), KS max 0,0034 (p > 0,17 partout), **adversarial validation AUC = 0,5008** (LightGBM 5 folds ; l'id seul donne 0,5011). Le test est un tirage aléatoire du même jeu.

**Doublons, corrélations, colonnes déduites**
- 0 ligne dupliquée (hors id/target) dans le train ou le test, 0 ligne du test identique à une ligne du train. Hors ps_calc : 4 lignes en trop (8 lignes impliquées, toutes cible 0) et 2 lignes test identiques à une ligne train : négligeable.
- **Aucune paire |r| > 0,9** entre variables. La plus forte : ps_ind_12_bin / ps_ind_14 (0,89), puis ps_reg_02 / ps_reg_03 (0,74), ps_car_12 / ps_car_13 (0,67).
- **Colonnes déduites (vérifié à 100 %)** : `ps_ind_14` = somme exacte de ps_ind_10..13_bin (redondante) ; `ps_ind_06..09_bin` sont un **one-hot** (la somme vaut toujours 1 : 4 modalités d'une seule variable) ; `ps_ind_16..18_bin` : somme 0 ou 1 jamais plus (one-hot avec une 4e modalité « aucune », 6,5 % des lignes). Les `ps_calc_15..20_bin` ne sont pas liées (sommes de 0 à 6).
- **Structure des continues** : `ps_car_15` = racine carrée d'un entier de 0 à 14 (15 valeurs) ; `ps_car_12` et `ps_car_14` = racine de (entier/10 000) ; `ps_reg_03` = racine de (entier/1 600) (vérifié à 100 % des valeurs) ; `ps_car_13` n'a pas de structure simple repérée (62 546 valeurs distinctes). `ps_reg_01/02` et `ps_calc_01..03` sont des multiples de 0,1. Ces racines carrées sont des transformations monotones : elles ne changent rien pour des arbres.
- **Codes spéciaux** : à part -1, rien de net. Les valeurs 7, 9, 77, 99 existent mais dans des ordinales/catégorielles où elles sont plausibles (ps_car_11_cat a 77 et 99 parmi ses 104 modalités ; ps_ind_15 a 7 et 9 sur une échelle 0..13). Aucune valeur négative autre que -1. Pas de valeur impossible. Outliers modérés (max/p99,9 < 2) : ps_reg_03 max 4,04, ps_car_12 max 1,26, ps_car_13 max 3,52. Détail : `audit_valeurs.csv`.

## Décisions prises et justification
- **Rien n'a été corrigé** (consigne). Les -1 sont conservés dans les parquet, car être manquant est informatif.
- Gini des catégorielles = moyenne des Gini par fold (et pas un Gini « poolé » sur les 5 folds) : le poolé est biaisé car la moyenne d'apprentissage change d'un fold à l'autre (mesuré : faux bruit de 0,05 sur des binaires). Corrigé en cours de route.
- Seuil de « signal réel » = quantile 95 % du maximum du |Gini| sous cible mélangée sur 57 variables (0,014), plus rigoureux qu'un seuil arbitraire.
- Les repères Kaggle (calc inutiles, car_13 meilleure variable, manquants informatifs) ont été **vérifiés** par nos propres mesures ci-dessus, et sont confirmés.

## Points d'attention / risques
- Le signal est faible : écart-type du Gini d'un fold ≈ 0,0075. Une amélioration de moins de ~0,005 est du bruit.
- Le Gini univarié ne voit pas les interactions : une variable à 0,02 seule peut servir dans un arbre (cf. ps_car_14 ou ps_ind_03 qui passent de 0,02 à 0,09 avec un encodage non monotone).
- ps_car_11_cat (104 modalités) : tout encodage cible hors des folds crée de la fuite.
- Les lifts des manquants rares (66 à 416 lignes) sont très incertains.
- La structure « racine de entier » de ps_reg_03/ps_car_12/ps_car_14 est un artefact d'anonymisation, ne pas l'interpréter métier.

## Recommandations pour la suite
1. ml-engineer : garder les -1 (NaN natif ou catégorie) et tester avec/sans les ps_calc_* sur les mêmes folds.
2. data-cleaner : ne rien imputer à la moyenne ; produire une version `v1` avec ajouts d'indicateurs et d'agrégats (liste ci-dessous), à comparer à la version brute.
3. Ne pas chercher de gain sur la dérive (il n'y en a pas) : la validation croisée classique est fiable pour estimer le score du test.
4. Pour l'oral : « on a mesuré avant de supposer » (calc, manquants, dérive, fuite).

## Fichiers produits
- Script : `usine/src/01_audit.py` ; sondes : `usine/src/01_audit_extra.py`, `usine/src/01_audit_extra2.py` ; logs : `usine/logs/01_audit_3.log`, `01_extra.log`, `01_extra2.log`.
- Données typées : `usine/data/interim/train_raw.parquet`, `usine/data/interim/test_raw.parquet`.
- Tableaux : `usine/rapports/audit_variables.csv` (profil + signal + dérive, une ligne par variable), `audit_derive.csv`, `audit_modalites.csv` (taux de cible par modalité), `audit_valeurs.csv`, `audit_correlations_fortes.csv`, `audit_dependances.csv` (vide : aucune dépendance fonctionnelle détectée hors cas listés), `audit_resume.json`.
- Figures (`usine/figures/`) : `01_cible.png`, `01_manquants.png`, `01_gini_top20.png`, `01_gini_par_groupe.png`, `01_manquants_vs_cible.png`, `01_correlations.png`, `01_derive.png`.

## Corrections / transformations possibles (à décider avec l'utilisateur)
1. **Manquants -1 : les garder, ajouter un indicateur « manquant » pour ps_car_03_cat, ps_car_05_cat, ps_reg_03, ps_car_07_cat, ps_ind_05_cat, plus le nombre de -1 par ligne.** Avantage : conserve et exploite le signal (Gini 0,07 / 0,06 / 0,04 pour les indicateurs). Risque : quasi nul avec des arbres (qui voient déjà le -1) ; redondance. Avis : oui pour le compte de -1 par ligne et les 3 gros manquants ; ne pas imputer à la moyenne.
2. **Retirer les 20 variables ps_calc_*.** Avantage : moins de bruit, entraînement plus rapide, modèle plus simple à défendre. Risque : perdre une interaction cachée (peu probable : calc seules = Gini 0,002). Gain mesuré en retirant : +0,0017 (dans le bruit). Avis : à tester en CV ; si le résultat est égal ou meilleur, les retirer.
3. **Supprimer `ps_ind_14`** (somme exacte de ps_ind_10..13_bin). Avantage : enlève une redondance. Risque : aucun (information reconstructible). Avis : oui, ou garder seulement ps_ind_14 et retirer les 4 binaires rares ; tester.
4. **Regrouper `ps_ind_06..09_bin` en une seule catégorielle (4 modalités) et `ps_ind_16..18_bin` en une catégorielle (4 modalités, dont « aucune »).** Avantage : fait ressortir l'effet « type » au lieu de 3-4 colonnes dépendantes ; utile pour les modèles linéaires. Risque : nul (information identique). Avis : utile pour linéaire/CatBoost ; neutre pour LightGBM.
5. **Encoder ps_car_11_cat (104 modalités) par le taux de cible dans les folds (lissé), ou la passer en catégorielle native CatBoost/LightGBM.** Avantage : son Gini passe de 0,026 (brut) à 0,135 (encodé), 2e variable du jeu. Risque : fuite si l'encodage est calculé sur tout le train ; modalités rares instables. Avis : oui, strictement dans les folds.
6. **Traiter toutes les `_cat` comme catégorielles (natif ou encodage cible/one-hot) et non comme des nombres.** Avantage : les valeurs 0..n ne sont pas ordonnées. Risque : surapprentissage sur les modalités rares ; one-hot gonfle les colonnes. Avis : oui, tester natif vs numérique.
7. **Regrouper les modalités très rares (ps_ind_04_cat -1 : 66 lignes, ps_car_01_cat -1 : 82, ps_car_02_cat -1 : 4, ps_car_11 -1 : 3, ps_car_12 -1 : 1).** Avantage : stabilité. Risque : on masque un éventuel vrai signal (lift x8 à x10 mais très incertain). Avis : ne pas imputer les 1-4 lignes ; garder les autres comme catégorie à part, laisser le modèle décider.
8. **Rendre « entières » les variables en racine carrée (ps_car_15², ps_car_12²×10 000, ps_car_14²×10 000, ps_reg_03²×1 600).** Avantage : valeurs ordinales lisibles, mieux adaptées à un modèle linéaire ou un réseau. Risque : aucun pour des arbres (transformation monotone, aucun effet) ; peut déformer un modèle linéaire. Avis : optionnel, faible priorité.
9. **Écrêter les valeurs extrêmes (ps_reg_03 > p99,9, ps_car_12, ps_car_13).** Avantage : stabilise un modèle linéaire ou un réseau. Risque : perte d'information sur des cas rares ; inutile pour les arbres. Avis : non pour les modèles à arbres.
10. **Supprimer les quelques doublons (4 lignes en trop hors calc).** Avantage : propreté. Risque : aucun mais gain nul. Avis : ne rien faire.
11. **Interactions candidates à tester (pistes Kaggle à vérifier en CV, pas des faits)** : ps_reg_01 × ps_reg_02, ps_reg_03 avec ps_car_13, combinaisons de catégorielles (ps_car_04 × ps_car_02). Avantage : signal d'interaction. Risque : surapprentissage, bruit ±0,0075 par fold. Avis : tester un petit nombre, garder seulement si le gain dépasse l'écart-type.
12. **Rééquilibrer la cible (sous-échantillonnage, poids).** Avantage : parfois utile pour certains modèles. Risque : pour une métrique AUC, le gain est rarement net et on perd des données. Avis : tester en dernier ; ne pas l'adopter par défaut.
