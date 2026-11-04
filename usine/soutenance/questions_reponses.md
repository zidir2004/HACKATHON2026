# Questions probables du jury et réponses courtes

Réponses écrites pour être dites à l'oral : 2 à 4 phrases, sans jargon. Le membre indiqué est celui qui répond en premier (voir `plan.md`).

**1. Pourquoi pas de seuil, de matrice de confusion, de précision ? (Membre 3)**
La métrique du sujet est le Gini, qui ne regarde que l'ordre des prédictions : un assuré est-il classé plus risqué qu'un autre ? Un seuil ne sert que si on doit décider « oui/non » ; avec 3,6 % de sinistres, la précision est trompeuse (« jamais de sinistre » donnerait 96 %). Nous donnons à la place les déciles et la courbe de gains, qui parlent à un assureur.

**2. Votre validation croisée est-elle fiable ? (Membre 3)**
Oui, pour trois raisons. Le test d'adversaire (un modèle qui essaie de distinguer train et test) donne AUC 0,50 : le test ressemble au train. Les plis sont stratifiés et figés pour toutes les expériences. Un relecteur indépendant a recalculé nos scores sans fuite. Réserve honnête : l'arrêt anticipé rend la CV un peu optimiste (0,0005 à 0,0008) et le score dépend du découpage (0,276 à 0,279 pour le premier modèle) ; nous annonçons donc environ 0,281-0,283.

**3. Pourquoi retirer les variables ps_calc si le gain n'est pas significatif ? (Membre 2)**
Le gain n'est pas la raison, c'est la simplification. Ces 20 variables ont un signal individuel quasi nul (Gini 0,003) et 8 % de l'importance du modèle pour 35 % des variables ; les retirer n'a pas dégradé le score (+0,0022 sur 4 plis sur 5, non significatif). À performance égale, on garde le modèle le plus simple et le plus facile à expliquer. Nous l'annonçons comme une simplification, pas comme un gain.

**4. Pourquoi un mélange de trois modèles ? N'est-ce pas de la complexité inutile ? (Membre 4)**
C'est le seul changement qui a un gain net : +0,0026 sur les 5 plis, et le mélange bat le meilleur de ses trois membres dans chaque pli. Les trois algorithmes (LightGBM, XGBoost, CatBoost) ne font pas les mêmes erreurs : la moyenne les atténue. Nous avons testé des poids optimisés : aucun gain, donc poids égaux, c'est le plus simple et le plus robuste.

**5. Avez-vous utilisé Kaggle ? (Membre 5)**
Oui, et nous le disons : le jeu est « Porto Seguro Safe Driver » (2017). Les connaissances publiques (par exemple « les ps_calc sont du bruit ») ont servi de pistes, jamais de preuves : chaque idée a été retestée dans notre protocole, et plusieurs ne se sont pas confirmées chez nous (encodage cible, indicateurs de manquants). [À confirmer par l'équipe avant l'oral : le code et les soumissions sont les nôtres, aucune soumission publique reprise.]

**6. Pourquoi n'avez-vous pas imputé les valeurs manquantes ? (Membre 1)**
Parce que le manque est une information : les assurés avec ps_car_07_cat manquant ont 7,9 % de sinistres contre 3,5 % pour les autres (×2,2). Les arbres savent lire la valeur -1. Nous avons testé des indicateurs de manquant en plus : aucun gain (−0,0009).

**7. Votre gain est de 0,002, l'écart entre plis est de 0,008 : comment pouvez-vous dire que c'est réel ? (Membre 3)**
Parce que nous ne comparons pas des moyennes mais des différences appariées sur les mêmes plis. L'écart entre plis est un bruit commun aux deux modèles qui s'annule quand on soustrait. Un gain est retenu seulement s'il gagne sur les 5 plis ou s'il dépasse 2 erreurs-types. Pour le mélange, le gain est positif sur 5 plis sur 5.

**8. Un Gini de 0,28, est-ce bon ? (Membre 5)**
C'est un score typique pour ce type de données : le sinistre est largement aléatoire. Il est utile pour classer : les 20 % d'assurés les plus risqués concentrent 37 % des sinistres et le taux passe de 1,5 % à 8,0 % entre le premier et le dernier décile. Il ne permet pas de prédire un individu : nous le disons clairement.

**9. À quoi servirait ce modèle en pratique ? (Membre 5)**
À segmenter : ajuster la tarification, repérer les profils à surveiller, cibler des actions de prévention. Un assuré du décile le plus risqué a environ 5 fois plus de chances de déclarer un sinistre qu'un assuré du décile le moins risqué. Pas à refuser un assuré sur la base du score seul.

**10. Que signifient vos variables ? Peut-on dire que ps_car_13 est la puissance du véhicule ? (Membre 2)**
Non, les variables sont anonymisées : nous ne pouvons pas l'affirmer. Nous avançons seulement des hypothèses prudentes (ind = assuré, car = véhicule, reg = zone). Ce que nous savons, c'est que ps_car_13 est la variable la plus informative : le taux de sinistre passe de 2,3 % à plus de 7 % quand elle augmente.

**11. Y a-t-il un risque de fuite de données (data leakage) ? (Membre 3)**
Nous avons cherché : aucun identifiant lié à la cible (Gini de l'id 0,0007), aucun doublon train/test, et les traitements qui utilisent la cible (encodage de ps_car_11_cat) sont faits à l'intérieur de chaque pli. Le relecteur indépendant a vérifié ce point sur le code.

**12. Pourquoi avoir testé tant d'idées qui ne marchent pas ? N'est-ce pas du temps perdu ? (Membre 4)**
C'est le résultat : savoir que les indicateurs de manquants, l'encodage cible, les réglages fins ne servent à rien nous a permis de nous arrêter et de consacrer le temps à l'analyse. Chaque test est tracé dans le registre avec son verdict. Un échec mesuré est une information.

**13. Comment avez-vous travaillé à cinq ? (Membre 5, ou tous)**
Rôles séparés : exploration des données, préparation, modèles, relecture indépendante, analyse et soutenance. Règle commune : un protocole unique, un journal où chaque décision est datée et justifiée, et une relecture par quelqu'un qui n'est pas l'auteur. À adapter avec les prénoms et ce que chacun a vraiment fait.

**14. Votre modèle est-il expliqué ? Pourquoi faire confiance à SHAP ? (Membre 4)**
SHAP mesure ce que le modèle utilise, pas une cause. Nous le recoupons avec l'analyse descriptive : les variables importantes pour le modèle (ps_car_13, ps_ind_03, ps_ind_05_cat) sont celles dont le taux de sinistre varie le plus dans les données brutes. Précision : SHAP a été calculé sur le modèle LightGBM E01, un des trois membres du mélange.

**15. Quel score attendez-vous sur le serveur ? (Membre 5)**
Environ 0,281 à 0,283 : la validation croisée donne 0,2834 mais elle est légèrement optimiste. Nous ne cherchons pas à sur-ajuster le classement public : nous ne soumettons que ce qui progresse en validation croisée.

**16. Les 3,6 % de sinistres : fallait-il rééquilibrer (sur-échantillonner) ? (Membre 1)**
Nous ne l'avons pas fait, par choix : le Gini ne dépend que de l'ordre et rééquilibrer fausse les probabilités (décision du journal, non testée expérimentalement : à dire honnêtement si on nous le demande). Nous avons stratifié les plis, pour que chaque pli contienne 3,6 % de sinistres.

**17. Pourquoi E06 (catégorielles natives) est-il plus mauvais ? (Membre 4)**
Hypothèse, non prouvée : avec 104 modalités dont certaines peu fréquentes et un signal faible, l'algorithme crée des groupes de modalités qui collent au bruit. Mesuré : −0,010 sur les 5 plis, c'est notre pire résultat.
