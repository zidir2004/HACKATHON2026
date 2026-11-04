# Plan de soutenance — modèle final 2_306 (E33f) — 12 slides, 14 minutes, 5 intervenants

Barème de l'oral (13 points) : méthodologie /4, analyse approfondie du sujet /4, clarté du code /2, travail en équipe /3.
Le professeur insiste : « pas seulement de la prédiction, aussi de l'analyse » : on montre donc les données, les distributions et l'usage métier, pas seulement le score.
Chiffres à connaître par cœur (tous du modèle final) : **score serveur 0,30079 ; Gini de validation 0,2853 ± 0,0074 ; baseline logistique 0,239 ; LightGBM seul 0,279 ; les 20 % d'assurés les plus risqués = 37,2 % des sinistres ; taux de sinistre ×5,5 entre le décile 1 (1,5 %) et le décile 10 (8,1 %).**

| Membre | Slides | Temps | Thème |
|---|---|---|---|
| Membre 1 | 1, 2, 3 | 3 min 00 | titre et sujet, données, audit |
| Membre 2 | 4, 5 | 2 min 15 | protocole de validation, baseline |
| Membre 3 | 6, 7 | 2 min 15 | itérations, ce qui a marché ou non, plafond des boostings |
| Membre 4 | 8, 9 | 2 min 45 | piste Kaggle réimplémentée, modèle final, démonstration du code |
| Membre 5 | 10, 11, 12 | 3 min 45 | résultats, analyse métier, limites et conclusion |
| **Total** | 12 | **14 min 00** | |

Phrase de passage : chaque membre termine par une phrase qui présente le suivant (« ... et c'est ce qui nous a conduits à définir notre protocole, Membre 2 »). La répartition est imposée par l'équipe ; le jury demande souvent « qui a fait quoi ? » : chacun sait dire en une phrase son rôle dans l'usine (lead-tech, data, nettoyage, modélisation, revue indépendante, analyse).

---

## Membre 1 — slides 1 à 3 (3 min)

### Slide 1 — Le sujet (Membre 1, 1 min)
- **Message clé** : on prédit si un assuré auto déclare un sinistre ; le signal est faible, donc le travail est autant de comprendre les données que de prédire.
- **Figure** : `usine/figures/01_cible.png` (déséquilibre : 3,64 % de sinistres).
- **Notes orales** :
  - Jeu public Porto Seguro (Kaggle 2017), on le dit dès le début.
  - 446 409 assurés d'entraînement, 148 803 de test, 57 variables anonymisées en 4 familles : ind (assuré), reg (région), car (véhicule), calc (calculées).
  - Métrique officielle : le Gini = 2 × AUC − 1. Il note l'ordre des assurés (qui est plus risqué qu'un autre), pas une décision oui/non.
  - Annoncer le fil rouge : « baseline 0,239, final 0,301 sur le serveur ».
  - Présenter l'équipe en une phrase : cinq rôles, un journal, une revue indépendante.

### Slide 2 — Les données (Membre 1, 1 min)
- **Message clé** : les données sont propres sur la forme mais piégeuses sur le fond : très déséquilibrées, avec des manquants qui parlent.
- **Figure** : `usine/figures/01_manquants_vs_cible.png` (en complément : `usine/figures/10_continues.png`, `usine/figures/10_categorielles.png`).
- **Notes orales** :
  - 3,64 % de sinistres : un modèle « jamais de sinistre » aurait 96 % de bonnes réponses, d'où le Gini et non la précision.
  - La valeur -1 signifie « manquant » et elle est informative : ps_car_07_cat manquant, 7,9 % de sinistres contre 3,5 % (×2,2). On n'a donc pas imputé.
  - Variables continues : plusieurs liens nets avec le risque (ps_car_13 : de 2,3 % à plus de 7 % de sinistres).
  - Catégorielles : le taux varie du simple au double selon la modalité, avec intervalles de confiance.

### Slide 3 — L'audit : trois constats (Membre 1, 1 min)
- **Message clé** : l'audit a validé la confiance dans la validation croisée et simplifié le jeu de variables.
- **Figure** : `usine/figures/10_signal_groupes.png` (+ `usine/figures/01_derive.png` si on veut montrer l'absence de dérive).
- **Notes orales** :
  - Constat 1, manquants informatifs : on les garde tels quels, les arbres savent les lire.
  - Constat 2, les 20 variables ps_calc sont du bruit : 35 % des variables mais 8 % de l'importance, signal individuel quasi nul. On les retire (simplification, effet mesuré +0,0022, non significatif).
  - Constat 3, pas de dérive : un modèle qui essaie de distinguer train et test échoue (AUC 0,50), donc le test ressemble au train.
  - Transition vers le Membre 2 : « comment mesurer proprement chaque idée ? ».

---

## Membre 2 — slides 4 et 5 (2 min 15)

### Slide 4 — Le protocole de validation (Membre 2, 1 min 15)
- **Message clé** : toutes les idées sont jugées avec le même protocole, par différences appariées, et rien n'est retenu sans règle ni relecture.
- **Figure** : schéma à main levée (5 plis figés, flèche « mêmes plis pour tous ») ; sinon montrer 5 lignes de `usine/experiences/comparaisons.csv`.
- **Notes orales** :
  - Validation croisée à 5 plis, graine 42, plis figés dans `experiences/folds.csv` : tous les modèles voient exactement les mêmes plis.
  - Un seul changement par expérience.
  - Comparaison appariée : on soustrait deux modèles pli par pli, ce qui annule le bruit commun (écart entre plis 0,007, gains de 0,001-0,003).
  - Règle : « garde » seulement si le gain est positif sur 5 plis sur 5 (ou grand devant son erreur-type) ; sinon « neutre » ou « rejeté ».
  - Revue indépendante : un relecteur recalcule les scores et cherche les fuites ; journal horodaté de toutes les décisions (`usine/JOURNAL.md`), registre de toutes les expériences.

### Slide 5 — La baseline (Membre 2, 1 min)
- **Message clé** : le grand saut vient du passage de la régression logistique aux arbres (+0,039) ; ensuite, il ne reste que de petits gains.
- **Figure** : `usine/figures/10_progression_modeles.png`.
- **Notes orales** :
  - E00, régression logistique : 0,239. E01, LightGBM sur données brutes : 0,279.
  - Pourquoi : des relations non linéaires et des interactions que la logistique ne voit pas.
  - Les barres d'erreur (0,008) sont plus grandes que les gains suivants : c'est pour cela qu'on compare pli par pli.
  - Transition vers le Membre 3 : « et ensuite, une trentaine d'idées testées ».

---

## Membre 3 — slides 6 et 7 (2 min 15)

### Slide 6 — Les itérations : ce qui a marché et ce qui n'a pas marché (Membre 3, 1 min 15)
- **Message clé** : sur des dizaines d'expériences, peu de gains confirmés ; les échecs mesurés nous ont dit où s'arrêter.
- **Figure** : `usine/figures/11_ce_qui_a_marche.png` (vert = gardé, gris = neutre, rouge = rejeté).
- **Notes orales** :
  - Ce qui a marché : le mélange de 3 algorithmes (+0,0026, 5 plis sur 5), un apprentissage plus lent (+0,0009), le mélange à 4 avec le réseau (+0,0019 vs 3 boostings).
  - Ce qui n'a pas marché : catégorielles natives (−0,010, notre pire résultat), 12 variables construites d'un coup (−0,003), paramètres bruts de Jahrer (−0,003).
  - Neutres : indicateurs de manquants, encodage cible, retrait d'une variable, graines multiples.
  - Leçon : les arbres gèrent déjà les manquants et les redondances ; ajouter des variables ajoute du bruit.

### Slide 7 — Le faux positif E27 et le plafond des boostings (Membre 3, 1 min)
- **Message clé** : un gain « 5 plis sur 5 » peut être de la chance ; nous l'avons démasqué, et les boostings plafonnent vers 0,283.
- **Figure** : `usine/figures/11_ce_qui_a_marche.png` (ligne E28) ; secours : tableau à écrire au tableau (E27 : graine 42 +0,0012 ; graine 7 −0,0004 ; graine 2026 +0,0005 ; moyenne +0,0005, 3 plis sur 5, neutre).
- **Notes orales** :
  - E27 (somme des variables binaires) paraissait gagnante sur une graine : nous avons refait avec 3 graines, le gain a disparu. Nouvelle règle : toute nouvelle variable doit être confirmée sur 3 graines.
  - Les 3 boostings (LightGBM, XGBoost, CatBoost) se ressemblent : corrélation de rang 0,94 à 0,96. Leur mélange plafonne à 0,283.
  - Pour progresser, il faut un modèle d'une autre nature : c'est la piste du Membre 4.
  - Transition : « la piste Kaggle, Membre 4 ».

---

## Membre 4 — slides 8 et 9 (2 min 45)

### Slide 8 — La piste Kaggle réimplémentée : autoencodeur et réseau (Membre 4, 1 min 15)
- **Message clé** : un réseau de neurones, plus faible seul, apporte de la diversité au mélange ; nous avons cité la piste publique, puis tout revalidé.
- **Figures** : `usine/figures/11_correlations_membres.png` et `usine/figures/11_apport_membres.png`.
- **Notes orales** :
  - Piste publique : la solution classée 1re de Porto Seguro (M. Jahrer), citée, réimplémentée en version réduite ; ses paramètres bruts étaient même moins bons chez nous (E29 : −0,003).
  - Autoencodeur débruiteur : un réseau apprend à reconstruire les données après avoir échangé 15 % des valeurs au hasard ; il ne voit jamais la cible (non supervisé). Un second réseau (4500-1000-1000) apprend le risque à partir de ses activations.
  - Seul, le réseau est le plus faible (0,275), mais ses prédictions sont différentes : corrélation de rang 0,91 avec les arbres contre 0,94 à 0,96 entre arbres.
  - Résultat : le mélange à 4 (0,2853) bat chacun de ses membres et le mélange à 3 (+0,0019, 5 plis sur 5).

### Slide 9 — Le modèle final 2_306 et le code (Membre 4, 1 min 30)
- **Message clé** : un pipeline simple à décrire (4 membres, moyenne de rangs, remise en probabilité) et un code lisible et reproductible.
- **Figure** : `usine/figures/11_schema_modele_final.png`.
- **Démonstration du code** (30 s, fichiers ouverts à l'avance) : `usine/src/03_train.py` (une expérience = une configuration, fonction de comparaison avec la règle), `usine/src/06_melange.py` (moyenne des rangs), `usine/src/07_rangs_vers_proba.py` (remise à l'échelle), `usine/src/13_nn_dae.py` (réseau). Ne pas ouvrir les `.bak`.
- **Notes orales** :
  - 4 membres : E30 LightGBM (paramètres Jahrer + one-hot), E07 XGBoost, E08 CatBoost, E32c réseau sur autoencodeur (3 graines).
  - Moyenne des rangs à poids égaux : des poids optimisés n'ont rien apporté, on garde le plus simple.
  - Remise à l'échelle pour obtenir des probabilités (taux moyen 3,65 %).
  - Code : noms en français, docstring avec la commande à lancer, résultats écrits en fichiers (registre, comparaisons, OOF), relancer = une ligne.
  - Transition : « les résultats, Membre 5 ».

---

## Membre 5 — slides 10 à 12 (3 min 45)

### Slide 10 — Résultats : 0,30079 sur le serveur (Membre 5, 1 min 15)
- **Message clé** : le modèle final progresse régulièrement et le serveur confirme avec 0,30079.
- **Figure** : `usine/figures/11_progression_soumissions.png`.
- **Notes orales** :
  - Gini de validation des 5 soumissions : 0,2787, 0,2834, 0,2843, 0,2853, 0,2856. Scores serveur officiels : 2_306 = 0,30079 ; 2_307 = 0,3003.
  - Pourquoi 2_306 et pas 2_307 : la validation donnait +0,0003 à 2_307, le serveur l'a mis un peu derrière : à égalité dans le bruit, on garde le plus simple (un réseau de moins).
  - Le serveur est au-dessus de la validation (0,301 contre 0,285) : variance d'échantillonnage du test (écart-type entre plis 0,007), pas une fuite (adversarial 0,50).
  - Rappeler la baseline 0,239.

### Slide 11 — Analyse métier : à quoi sert le modèle (Membre 5, 1 min 30)
- **Message clé** : le modèle classe les assurés : les 20 % les plus risqués concentrent 37,2 % des sinistres et le taux est multiplié par 5,5 entre le décile 1 et le décile 10.
- **Figures** : `usine/figures/11_gains_deciles_E33f.png` ; SHAP : `usine/figures/05_shap_importance.png` et `usine/figures/05_shap_dep_ps_car_13.png` ; profils : `usine/figures/05_profils.png`.
- **Notes orales** :
  - Déciles : 1,5 % de sinistres dans le décile le moins risqué, 8,1 % dans le plus risqué (moyenne 3,64 %).
  - Usages : segmentation tarifaire, priorisation de contrôles, prévention ciblée ; pas de refus d'un assuré sur le score seul.
  - SHAP (calculé sur le LightGBM de la baseline, mêmes variables que le modèle final) : ps_car_13, ps_ind_03, ps_ind_05_cat, ps_ind_15, les mêmes que l'analyse descriptive : pas de raccourci étrange. Les ps_calc comptent peu.
  - Variables anonymisées : hypothèses prudentes seulement (car = véhicule, reg = zone).

### Slide 12 — Limites et conclusion (Membre 5, 1 min)
- **Message clé** : un modèle honnête, utile pour classer, limité par les données ; démarche rigoureuse et reproductible.
- **Figure** : aucune (ou `usine/figures/11_apport_membres.png` en rappel).
- **Notes orales** :
  - Limites : AUC 0,64 environ, le sinistre est largement aléatoire ; variables anonymisées ; pas de date (pas de test temporel) ; l'autoencodeur est entraîné aussi sur le test, sans cible (technique standard, à dire avant qu'on le demande) ; gains récents du même ordre que le bruit.
  - Pistes : variables métier avec un expert, données historiques client.
  - Conclusion (10 s, tous) : « Cinq personnes, un protocole, un journal, une revue indépendante : chaque chiffre annoncé a été recalculé par quelqu'un d'autre que son auteur. »

---

## Récit des pistes qui n'ont pas marché (à raconter si on a 20 secondes de plus)
| Piste | Résultat | Ce que ça a appris |
|---|---|---|
| Catégorielles natives (E06) | −0,010, 0 plis sur 5 | avec 104 modalités et un signal faible, l'algorithme colle au bruit ; le one-hot est meilleur |
| 12 variables construites (E19) | −0,003 | les arbres trouvent déjà les interactions ; trop de variables = bruit |
| E27 somme des binaires | +0,0012 sur une graine, +0,0005 sur 3 graines (3/5) | un gain 5/5 peut être de la chance : toujours 3 graines |
| Indicateurs de manquants (E03), encodage cible (E05) | neutres | les arbres lisent déjà le -1 |
| Paramètres bruts de Jahrer (E29) | −0,003 | une recette publique ne se transpose pas telle quelle ; il a fallu ajouter le one-hot (E30b) |
| Autoencodeur « repris » (E41) | −0,0002, 1 pli sur 5 | plus de réseau ne veut pas dire mieux |
| Réseau sans autoencodeur (E52) | 0,265, neutre dans le mélange | c'est l'autoencodeur qui donne la bonne représentation |
| 2_307 (2 réseaux) | validation +0,0003, serveur 0,3003 < 0,30079 | à égalité dans le bruit : on garde le plus simple |

## Liste des figures et leur message en une phrase
| Figure | Slide | Message |
|---|---|---|
| `usine/figures/01_cible.png` | 1 | Seulement 3,64 % de sinistres : la précision serait trompeuse. |
| `usine/figures/01_manquants_vs_cible.png` | 2 | Un manquant est une information : le taux de sinistre double. |
| `usine/figures/10_signal_groupes.png` | 3 | Le signal est dans ind, reg, car ; les ps_calc sont du bruit. |
| `usine/figures/10_progression_modeles.png` | 5 | La logistique à LightGBM est le grand saut (+0,039). |
| `usine/figures/11_ce_qui_a_marche.png` | 6, 7 | Peu de gains confirmés ; les échecs sont mesurés et nous guident. |
| `usine/figures/11_correlations_membres.png` | 8 | Le réseau est différent des arbres (0,91 contre 0,94-0,96). |
| `usine/figures/11_apport_membres.png` | 8 | Le mélange bat chacun de ses membres, même avec un réseau plus faible. |
| `usine/figures/11_schema_modele_final.png` | 9 | Le pipeline de 2_306 en une image. |
| `usine/figures/11_progression_soumissions.png` | 10 | Le score progresse, le serveur confirme (0,30079). |
| `usine/figures/11_gains_deciles_E33f.png` | 11 | Les 20 % les plus risqués = 37,2 % des sinistres ; ×5,5 entre déciles. |
| `usine/figures/05_shap_importance.png`, `05_shap_dep_ps_car_13.png`, `05_profils.png` | 11 | Le modèle utilise les variables que l'analyse descriptive désigne. |

## Préparation d'ici 16h30
Ordre du jour : 14h30-15h10 chacun relit ses slides et ses chiffres ; 15h10-15h40 première répétition complète chronométrée (14 min, passages compris) ; 15h40-16h10 corrections et questions croisées ; 16h10-16h30 deuxième répétition courte, ouvrir les fichiers de code, tester le partage d'écran.

| Membre | À faire | 2 questions que le jury lui posera probablement |
|---|---|---|
| Membre 1 | Relire les slides 1-3 ; savoir expliquer le Gini, -1 = manquant, 3,64 %, le test d'adversaire (0,50) ; préparer la phrase de présentation d'équipe. | « Pourquoi n'avoir pas imputé les manquants ? » (Q6) ; « Fallait-il rééquilibrer les classes ? » (Q16) |
| Membre 2 | Relire les slides 4-5 ; savoir expliquer la comparaison appariée avec un exemple chiffré (E09 : +0,0026 sur 5 plis sur 5) ; connaître 0,239 et 0,279. | « Votre validation est-elle fiable ? » (Q2) ; « Pourquoi retirer les ps_calc ? » (Q3) |
| Membre 3 | Relire les slides 6-7 ; savoir raconter E27 en 20 s (graines 42, 7, 2026) ; connaître les 3 pires échecs. | « Gain de 0,002 avec un écart de 0,008, est-ce réel ? » (Q7) ; « Pourquoi tester tant de choses qui échouent ? » (Q12) |
| Membre 4 | Relire les slides 8-9 ; savoir expliquer l'autoencodeur en 3 phrases sans jargon ; ouvrir les fichiers de code à l'avance ; savoir répondre sur Kaggle. | « Avez-vous copié Kaggle ? » (Q5) ; « Pourquoi l'autoencodeur voit-il le test ? » (Q8) |
| Membre 5 | Relire les slides 10-12 ; connaître 0,30079, 0,2853, 37,2 %, 1,5 % / 8,1 %, ×5,5 ; savoir justifier 2_306 contre 2_307 et l'écart serveur / validation. | « Pourquoi 2_306 et pas 2_307 ? » (Q9) ; « Serveur 0,301 > validation 0,285 ? » (Q10) |

Tous : savoir dire en une phrase « qui a fait quoi » et connaître les phrases de passage ; si une question dépasse son périmètre, le membre concerné (voir `usine/soutenance/questions_reponses.md`) prend le relais.
