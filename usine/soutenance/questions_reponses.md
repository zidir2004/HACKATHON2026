# Questions probables du jury et réponses courtes — modèle final 2_306 (E33f)

Réponses écrites pour être dites à l'oral : 2 à 4 phrases, sans jargon. Le membre indiqué répond en premier (voir `usine/soutenance/plan.md`). Chiffres de référence : serveur 0,30079 ; validation 0,2853 ± 0,0074 ; baseline 0,239 ; 20 % les plus risqués = 37,2 % des sinistres ; taux 1,5 % à 8,1 % (×5,5).

**1. Pourquoi pas de seuil, de matrice de confusion, de précision ? (Membre 5)**
La métrique du sujet est le Gini, qui ne regarde que l'ordre des prédictions : un assuré est-il classé plus risqué qu'un autre ? Un seuil ne sert que pour une décision oui/non ; avec 3,64 % de sinistres, la précision est trompeuse (« jamais de sinistre » donnerait 96 %). Nous donnons à la place les déciles et la courbe de gains, qui parlent à un assureur.

**2. Votre validation croisée est-elle fiable ? (Membre 2)**
Oui, pour trois raisons. Le test d'adversaire (un modèle qui essaie de distinguer train et test) donne AUC 0,50 : le test ressemble au train. Les 5 plis sont stratifiés et figés, identiques pour toutes les expériences. Un relecteur indépendant a recalculé nos scores et cherché les fuites. Réserve honnête : le score varie d'un pli à l'autre (écart-type 0,007).

**3. Pourquoi retirer les variables ps_calc si le gain n'est pas significatif ? (Membre 2)**
La raison est la simplification, pas le gain. Ces 20 variables ont un signal individuel quasi nul et 8 % de l'importance du modèle pour 35 % des variables ; les retirer n'a pas dégradé le score (+0,0022, 4 plis sur 5). À performance égale, on garde le modèle le plus simple. Nous l'annonçons comme une simplification, pas comme un gain.

**4. Pourquoi un mélange de quatre modèles ? N'est-ce pas de la complexité inutile ? (Membre 4)**
Le mélange est le seul changement qui donne des gains nets et répétés : 3 boostings +0,0026 (5 plis sur 5), puis le réseau +0,0019 de plus (5 plis sur 5). Les modèles ne font pas les mêmes erreurs, la moyenne les atténue. Les poids optimisés n'ont rien apporté : poids égaux, c'est le plus simple et le plus robuste.

**5. Avez-vous copié Kaggle ? (Membre 4)**
Non. Nous citons la piste publique (la solution classée 1re de Porto Seguro) et nous la disons ouvertement : nous l'avons réimplémentée en version réduite, sur notre propre code, et validée par notre protocole. Ses paramètres bruts étaient même moins bons chez nous (E29 : −0,003) ; il a fallu les adapter (one-hot des catégorielles). Aucune soumission publique n'a été reprise ; chaque idée reprise a été testée comme les autres.

**6. Pourquoi n'avez-vous pas imputé les valeurs manquantes ? (Membre 1)**
Parce que le manque est une information : ps_car_07_cat manquant donne 7,9 % de sinistres contre 3,5 % (×2,2). Les arbres savent lire la valeur -1. Nous avons testé des indicateurs de manquant en plus : aucun gain (−0,0009).

**7. Votre gain est de 0,002, l'écart entre plis est de 0,008 : comment pouvez-vous dire que c'est réel ? (Membre 3, relais Membre 2)**
Nous ne comparons pas des moyennes mais des différences appariées sur les mêmes plis : l'écart entre plis est un bruit commun aux deux modèles qui s'annule quand on soustrait. Un gain est retenu seulement s'il est positif sur les 5 plis. Le mélange à 4 gagne sur les 5 plis. Et nous avons appris à nous méfier : E27 gagnait sur 5 plis avec une graine, mais pas avec 3 graines.

**8. Pourquoi l'autoencodeur voit-il le test ? N'est-ce pas de la triche ? (Membre 4)**
L'autoencodeur est non supervisé : il n'utilise jamais la cible, il apprend seulement à reconstruire les variables, sur train et test. C'est une technique standard (apprentissage « transductif ») ; le relecteur a vérifié que la cible est retirée avant. Le réseau qui apprend le risque, lui, est entraîné pli par pli avec les plis figés.

**9. Pourquoi 2_306 et pas 2_307 ? (Membre 5)**
2_307 (deux réseaux) avait +0,0003 en validation mais a fait 0,3003 sur le serveur contre 0,30079 pour 2_306. Les deux sont à égalité dans le bruit (écart-type 0,007 entre plis). À égalité, on garde le plus simple : un réseau de moins, moins de risque. Nous ne choisissons pas sur un seul score serveur : ce serait du sur-ajustement au classement public.

**10. Le serveur donne 0,301 mais votre validation 0,285 : y a-t-il une fuite ? (Membre 5, relais Membre 2)**
Non. L'écart de +0,015 correspond à environ 2 écarts-types entre plis (0,007) : c'est la variance d'échantillonnage du jeu de test, un test de 148 803 assurés avec 3,6 % de sinistres. Une fuite se verrait par un test qui diffère du train : l'adversarial validation donne 0,50. Le score serveur est une seule mesure bruitée, pas une preuve de supériorité ; la validation sur 446 409 assurés est plus stable.

**11. Y a-t-il un risque de fuite de données (data leakage) ? (Membre 2)**
Nous avons cherché : aucun identifiant lié à la cible (Gini de l'id 0,0007), aucun doublon train/test, traitements utilisant la cible faits à l'intérieur de chaque pli, autoencodeur sans cible. Le relecteur indépendant a vérifié ce point sur le code (rapports 92 et 93).

**12. Pourquoi avoir testé tant d'idées qui ne marchent pas ? (Membre 3)**
C'est le résultat : savoir que les indicateurs de manquants, l'encodage cible, les variables construites ou les réglages fins ne servent à rien nous a fait choisir la seule piste utile, un modèle d'une autre nature. Chaque test est tracé dans le registre avec son verdict. Un échec mesuré est une information.

**13. Un Gini de 0,285, est-ce bon ? (Membre 5)**
C'est un score typique pour ce type de données : le sinistre est largement aléatoire (AUC 0,64). Il est utile pour classer : les 20 % d'assurés les plus risqués concentrent 37,2 % des sinistres, et le taux passe de 1,5 % à 8,1 %. Il ne permet pas de prédire un individu : nous le disons clairement.

**14. À quoi servirait ce modèle en pratique ? (Membre 5)**
À segmenter : ajuster la tarification, repérer des profils à surveiller, cibler la prévention. Un assuré du décile le plus risqué déclare un sinistre environ 5,5 fois plus souvent qu'un assuré du décile le moins risqué. Pas à refuser un assuré sur le score seul.

**15. Que signifient vos variables ? Peut-on dire que ps_car_13 est la puissance du véhicule ? (Membre 1)**
Non, les variables sont anonymisées : nous ne pouvons pas l'affirmer. Nous avançons seulement des hypothèses prudentes (ind = assuré, car = véhicule, reg = zone). Ce que nous savons : ps_car_13 est la variable la plus informative, le taux de sinistre passe de 2,3 % à plus de 7 % quand elle augmente.

**16. Les 3,64 % de sinistres : fallait-il rééquilibrer ? (Membre 1)**
Nous ne l'avons pas fait, par choix : le Gini ne dépend que de l'ordre et rééquilibrer fausse les probabilités (décision du journal, non testée : à dire honnêtement). Nous avons stratifié les plis pour que chacun contienne 3,64 % de sinistres. À la fin, une remise à l'échelle redonne des probabilités réalistes (taux moyen 3,65 %).

**17. Comment avez-vous travaillé à cinq ? (Membre 3, tous)**
Rôles séparés : exploration des données, préparation, modèles, relecture indépendante, analyse et soutenance. Règle commune : un protocole unique, un journal où chaque décision est datée et justifiée, une relecture par quelqu'un qui n'est pas l'auteur. À adapter avec les prénoms et ce que chacun a vraiment fait.

**18. Votre modèle est-il expliqué ? Pourquoi faire confiance à SHAP ? (Membre 5)**
SHAP mesure ce que le modèle utilise, pas une cause. Nous le recoupons avec l'analyse descriptive : les variables importantes (ps_car_13, ps_ind_03, ps_ind_05_cat) sont celles dont le taux de sinistre varie le plus dans les données brutes. Précision honnête : SHAP a été calculé sur le LightGBM de la baseline (mêmes variables que le modèle final), pas sur le mélange complet.

**19. Pourquoi un réseau, plus faible seul, améliore-t-il le mélange ? (Membre 4)**
Parce qu'il se trompe autrement : sa corrélation de rang avec les arbres est de 0,91 contre 0,94 à 0,96 entre arbres. En moyennant, les erreurs différentes se compensent. C'est comme demander l'avis d'un expert d'un autre domaine.

**20. Pourquoi E06 (catégorielles natives) est-il plus mauvais ? (Membre 3)**
Hypothèse, non prouvée : avec 104 modalités dont certaines rares et un signal faible, l'algorithme crée des groupes de modalités qui collent au bruit. Mesuré : −0,010 sur les 5 plis, notre pire résultat.

**21. Quelle est la différence entre moyenne des rangs et moyenne des probabilités ? (Membre 4)**
Chaque modèle produit des probabilités sur des échelles différentes ; on les transforme en rangs (1er, 2e, ...) avant de moyenner, ce qui est équitable et suffit pour le Gini. Une dernière étape remet le résultat à l'échelle d'une probabilité pour le fichier de soumission.

**22. Que feriez-vous avec plus de temps ? (Membre 5)**
Construire des variables avec un expert métier, tester des données de date ou d'historique client si elles existent, et vérifier la stabilité du modèle dans le temps. Nous ne ferions pas plus de mélanges : nous avons atteint le plafond de ce que la validation peut distinguer.
