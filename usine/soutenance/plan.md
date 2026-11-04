# Plan de soutenance — 14 minutes, 12 slides, 5 intervenants

Barème de l'oral (13 points) : méthodologie /4, analyse approfondie du sujet /4, clarté du code /2, travail en équipe /3.
Le prof insiste : « pas seulement de la prédiction, aussi de l'analyse : distributions des données, des variables ».
Donc : **5 minutes sur les données et leurs distributions (slides 1-5), 4 minutes sur la méthode et la démarche (6-8)**, 3 minutes sur le modèle et son usage (9-10), 2 minutes sur le code et les limites (11-12).
Chaque membre parle au moins 2 min 30 et présente une partie qu'il sait défendre en questions. **La répartition est une proposition : à ajuster selon qui a réellement fait quoi (le jury demande souvent « qui a fait quoi ? »).**

| Membre | Parties | Temps | Sujet de questions à prendre |
|---|---|---|---|
| Membre 1 | Slides 1-2 : sujet, données, qualité | 2 min | données, déséquilibre, manquants, absence de dérive |
| Membre 2 | Slides 3-5 : distributions, catégorielles, signal par groupe | 3 min 15 | analyse des variables, lecture métier |
| Membre 3 | Slides 6-7 : méthodologie, progression | 2 min 30 | protocole de validation, fiabilité de la CV |
| Membre 4 | Slides 8-9 : ce qui a marché ou non, explicabilité | 2 min 45 | choix de modèles, mélange, SHAP |
| Membre 5 | Slides 10-12 : performance, code, limites, conclusion | 3 min 30 | usage métier, code, limites, Kaggle |

Idée de mise en scène pour le travail d'équipe : chaque membre termine par une phrase de passage vers le suivant (« ... et c'est ce qui nous a conduits à définir notre protocole, Membre 3 »). Dire une fois à voix haute le fonctionnement d'équipe : lead-tech, data, nettoyage, modélisation, revue indépendante, analyse ; toutes les décisions sont dans un journal.

---

## Slide 1 — Le sujet et les données (Membre 1, 1 min)
- **Message** : on prédit si un assuré auto déclare un sinistre, mais le vrai enjeu est de comprendre les données, car le signal est faible.
- **Contenu** : 446 409 assurés d'entraînement, 148 803 de test, 57 variables anonymisées en 4 familles (ind = assuré, reg = région, car = véhicule, calc = variables calculées) ; métrique Gini = 2·AUC−1 ; jeu public Porto Seguro (Kaggle 2017), dit ouvertement.
- **Figure** : `usine/figures/01_cible.png` (déséquilibre 3,64 % de sinistres).

## Slide 2 — Qualité des données : trois constats (Membre 1, 1 min)
- **Message** : les données sont propres sur la forme mais piégeuses sur le fond : déséquilibre, manquants qui parlent, pas de dérive.
- **Contenu** : 3,64 % de positifs (donc un modèle « jamais de sinistre » aurait 96 % de bonnes réponses : d'où le Gini) ; -1 = manquant, informatif (ps_car_07_cat manquant : 7,9 % de sinistres contre 3,5 %) ; train et test identiques (test d'adversaire AUC 0,50) donc la validation croisée est fiable.
- **Figure** : `usine/figures/01_manquants_vs_cible.png`.

## Slide 3 — Distributions des variables continues (Membre 2, 1 min 15)
- **Message** : plusieurs variables montrent un lien net et progressif avec le risque (ps_car_13, ps_reg_02/03), d'autres un lien irrégulier (ps_ind_03).
- **Contenu** : ps_car_13 : le taux de sinistre passe de 2,3 % à plus de 7 % ; ps_reg_03 de 2,9 % à 5,9 % ; ps_ind_03 : 6,3 % pour la valeur 0, puis un creux à 3 % ; ps_reg_01 presque plat. Distributions asymétriques, valeurs discrètes (racines d'entiers).
- **Figure** : `usine/figures/10_continues.png`.

## Slide 4 — Catégorielles, binaires et manquants (Membre 2, 1 min 15)
- **Message** : le taux de sinistre varie du simple au double selon la modalité, et on le montre avec les effectifs et les intervalles de confiance, donc sans sur-interpréter les petits groupes.
- **Contenu** : ps_ind_17_bin = 1 : 5,5 % contre 3,4 % ; ps_ind_06_bin = 1 : 2,9 % contre 4,1 % ; ps_car_07_cat et ps_ind_05_cat manquants : taux ×2,2 ; modalités rares (100 assurés) : intervalle énorme, on ne conclut pas.
- **Figure** : `usine/figures/10_categorielles.png`. Complément si question : `usine/figures/05_modalites_ps_car_11_cat.png` (104 modalités, 2 % à 7,4 %).

## Slide 5 — Où est le signal ? (Membre 2, 45 s)
- **Message** : le signal est dans les familles ind, reg et car ; les 20 variables calc sont du bruit, ce qu'on a vérifié et pas seulement supposé.
- **Contenu** : calc = 35 % des variables mais 8 % de l'importance SHAP, signal univarié médian 0,003 contre 0,076 pour car ; l'effet du retrait des calc est mesuré (E02 : +0,0022, non significatif, donc simplification sans perte).
- **Figure** : `usine/figures/10_signal_groupes.png`.

## Slide 6 — Méthodologie : un protocole unique et vérifiable (Membre 3, 1 min 30)
- **Message** : toutes les expériences sont comparables parce qu'elles partagent le même protocole, et chaque décision est jugée sur des différences appariées.
- **Contenu** : StratifiedKFold 5 plis, graine 42, plis figés dans `experiences/folds.csv` ; un seul changement par expérience ; Gini = 2·AUC−1 par pli, moyenne ± écart-type, puis **différence appariée pli par pli avec la référence** ; règle de verdict (« garde » seulement si 5/5 plis gagnés ou gain > 2 erreurs-types) ; tout traitement appris (encodage cible) fait dans les plis ; revue indépendante (un reviewer recalcule les scores et cherche les fuites) ; journal horodaté de toutes les décisions (`JOURNAL.md`, registre de 30 expériences).
- **Figure** : schéma à dessiner sur la slide (5 plis, flèche « mêmes plis pour tous ») ; on peut montrer la table `usine/experiences/registre.csv`.

## Slide 7 — La progression des modèles (Membre 3, 1 min)
- **Message** : le grand saut vient du passage de la régression logistique aux arbres (+0,039) ; ensuite, les gains sont de l'ordre de 0,002 et seul le mélange est net.
- **Contenu** : E00 0,239, E01 0,279, E02 0,281, E09 0,283 ; E06 en rouge = échec instructif. Barres d'erreur = variation entre plis (0,008), bien plus grande que les gains : c'est pourquoi on compare pli par pli.
- **Figure** : `usine/figures/10_progression_modeles.png`.

## Slide 8 — Ce qui a marché et ce qui n'a pas marché (Membre 4, 1 min 30)
- **Message** : sur environ 30 expériences, 2 gains confirmés (mélange, pas d'apprentissage fin) et 1 piste prometteuse non confirmée ; les échecs nous ont appris autant que les succès.
- **Contenu** : réussite nette : le mélange de 3 algorithmes (+0,0026, 5/5 plis), pas d'apprentissage plus fin (+0,0009, 5/5), somme des binaires ps_ind (E27 : +0,0012 sur 5/5 plis avec une graine, mais seulement +0,0005, 3/5, avec 3 graines : non confirmée) ; échecs : catégorielles natives (−0,010), 12 variables construites d'un coup (−0,003, dont la plupart sont neutres seules) ; neutres : indicateurs de manquants, encodage cible, réglages, graines, poids du mélange. Leçon : les arbres gèrent déjà les manquants et les redondances ; trop de variables ajoute du bruit.
- **Figure** : `usine/figures/10_ce_qui_a_marche.png`. Récit détaillé : voir `usine/soutenance/questions_reponses.md` et le rapport 10.

## Slide 9 — Que regarde le modèle ? (Membre 4, 1 min 15)
- **Message** : le modèle s'appuie sur les mêmes variables que l'analyse descriptive (ps_car_13, ps_ind_03, ps_ind_05_cat, ps_ind_15) : il n'a pas trouvé un raccourci étrange.
- **Contenu** : importance SHAP (calculée sur le modèle LightGBM E01, qui est un des trois membres du mélange final et utilise les mêmes variables) ; effet de ps_car_13 croissant, cohérent avec le tableau de la slide 3 ; hypothèses métier prudentes (variables anonymisées) : ps_car_13 pourrait refléter une caractéristique du véhicule (valeur, puissance), ps_reg_ une zone géographique.
- **Figures** : `usine/figures/05_shap_importance.png`, `usine/figures/05_shap_dep_ps_car_13.png`.

## Slide 10 — Performance et usage métier (Membre 5, 1 min 15)
- **Message** : le modèle ne prédit pas un individu, il classe une population : les 20 % d'assurés les plus risqués concentrent 37 % des sinistres, et le taux est multiplié par 5,4 entre le décile le moins risqué et le plus risqué.
- **Contenu** : Gini 0,283 (AUC 0,64) ; décile 1 : 1,5 % de sinistres, décile 10 : 8,0 % ; usages : segmentation tarifaire, priorisation des contrôles, ciblage de prévention. Honnêteté : les deux populations se chevauchent beaucoup.
- **Figures** : `usine/figures/10_gains_E09.png`, `usine/figures/10_deciles_E09.png` (secours : `usine/figures/10_scores_par_classe.png`).

## Slide 11 — Le code : lisible et reproductible (Membre 5, 1 min)
- **Message** : un petit nombre de fichiers numérotés dans l'ordre du raisonnement, chacun avec son mode d'emploi en tête.
- **À montrer** : `usine/src/03_train.py` (1 expérience = 1 configuration ; la fonction `comparer` contient la règle de verdict ; docstring qui décrit les 5 étapes), `usine/src/utils.py` (métrique Gini, folds figés, registre, contrôle du format de soumission), `usine/src/06_melange.py` (mélange par moyenne de rangs). Pourquoi lisible : noms en français, une docstring d'en-tête avec la commande à lancer, pas de notebook, résultats écrits dans des fichiers (OOF, registre, comparaisons) ; relancer = une ligne.
- **Ne pas montrer** : les `.bak` et les scripts d'audit en détail.

## Slide 12 — Limites, pistes et conclusion (Membre 5, 1 min)
- **Message** : un modèle honnête : mieux que le hasard et utile pour classer, mais limité par les données ; la démarche est rigoureuse et reproductible.
- **Limites** : AUC 0,64 (le sinistre est largement aléatoire) ; variables anonymisées donc interprétation prudente ; pas de date donc pas de test temporel ; 3,6 % de positifs ; CV légèrement optimiste (+0,0005 à +0,0008, arrêt anticipé) : on annonce 0,281-0,283, pas 0,2834.
- **Pistes** : variables supplémentaires construites avec un expert (E27 prometteuse mais non confirmée), modèle de nature différente (réseau de neurones) pour diversifier le mélange, données de date/historique client si disponibles.
- **Conclusion (10 s, Membre 5 ou tous)** : « Cinq personnes, un protocole, un journal, une revue indépendante : chaque chiffre que nous annonçons a été recalculé par quelqu'un d'autre que son auteur. »

---

## Liste des figures et message en une phrase
| Figure | Message |
|---|---|
| `01_cible.png` | Seulement 3,6 % de sinistres : l'exactitude ne veut rien dire, on utilise le Gini. |
| `01_manquants_vs_cible.png` | Un manquant n'est pas un trou à combler : il porte de l'information. |
| `10_continues.png` | Le risque monte avec ps_car_13, ps_reg_02 et ps_reg_03 ; ps_ind_03 a une forme irrégulière. |
| `10_categorielles.png` | Des écarts du simple au double selon les modalités, avec des intervalles de confiance pour ne pas sur-interpréter. |
| `10_signal_groupes.png` | Le signal est dans ind, reg et car ; les ps_calc sont du bruit. |
| `10_progression_modeles.png` | Les arbres apportent +0,039 ; le reste ne se joue plus qu'au millième. |
| `10_ce_qui_a_marche.png` | 2 gains confirmés, des échecs clairs et beaucoup d'idées neutres : chaque idée a été mesurée. |
| `05_shap_importance.png` | Le modèle utilise surtout ps_car_13 et des variables de l'assuré. |
| `05_shap_dep_ps_car_13.png` | Plus ps_car_13 est élevée, plus le risque est élevé, comme dans les données brutes. |
| `10_gains_E09.png` | Les 20 % les plus risqués selon le modèle concentrent 37 % des sinistres. |
| `10_deciles_E09.png` | Le taux de sinistre passe de 1,5 % à 8,0 % du décile 1 au décile 10. |
| `10_scores_par_classe.png` | Les assurés avec et sans sinistre se chevauchent : on classe, on ne tranche pas. |

## Récit des pistes qui n'ont pas marché (à raconter en 30 secondes)
1. **Catégorielles natives (E06, −0,010, 0/5 plis)** : on pensait bien faire en laissant l'algorithme traiter les catégories ; il sur-apprend sur les modalités rares. Leçon : sur peu de signal, un traitement « plus fin » n'est pas forcément meilleur.
2. **Indicateurs de manquants (E03, −0,0009)** : on croyait aider le modèle ; les arbres voyaient déjà le -1. Leçon : la valeur -1 suffit, le manque est déjà lu.
3. **Encodage cible de ps_car_11_cat (E05, +0,0003)** : l'analyse montrait 104 modalités de 2 % à 7,4 %, donc l'idée semblait évidente ; gain nul. Leçon : une variable informative n'est pas une variable sous-exploitée.
4. **Réglages (E10-E16), graines (E17), poids du mélange (E18)** : tous sous 0,0005. Leçon : la variance vient des données (0,008 entre plis), pas du modèle ; on arrête de régler et on passe à la soutenance.
5. **12 variables construites (E19, −0,003)** : seule la somme des binaires ps_ind (E27) ressort (+0,0012) mais le gain fond avec 3 graines (+0,0005, 3/5) et dans le mélange (E24, +0,0001). Leçon : tester chaque variable seule ; en ajouter beaucoup à la fois détériore.
6. **Soutien Kaggle** : les repères du concours (ps_calc = bruit) ont servi de pistes, tous vérifiés par nous en validation croisée.
