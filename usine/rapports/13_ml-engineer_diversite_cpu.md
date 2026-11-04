# 13 - ML engineer (piste A) : diversité CPU autour de E33f (E60-E67) et ménage E32A

## 1. Ménage (demande du reviewer)
- `usine/src/11_regen_E32A.py` régénère `oof/E32A_{oof,test,folds_detail}.csv`, c'est-à-dire la moyenne des probabilités de
  E30b, E30bs7 et E30bs2026. Il ne touche ni à `oof/E32_*` (réseau de la piste B) ni au registre.
  Résultat : 0,28052 ± 0,00769, OOF 0,28039, identique à l'original.
- Commande reproductible de 2_305 : `python -u usine/src/11_regen_E32A.py` puis
  `python -u usine/src/07_rangs_vers_proba.py --exp E34 --membres E32A E07 E08 --sortie usine/submissions/2_305.csv`.
  Vérifié : la régénération (`submissions/_check_2_305_regen.csv`) est **identique octet par octet** (`cmp`) à 2_305.csv.
- `comparaisons.csv` : les lignes `E32,E29m` et `E33,E29m` de la piste A sont devenues `E32A,E29m` et `E33A,E29m`, et `E36,E32` est devenue `E36,E32A`.
  Les lignes E32/E33 restantes sont celles de la piste B.

## 2. Hypothèse
Des modèles construits autrement (DART, seuils aléatoires, modèle linéaire, forêt) seraient assez peu corrélés aux
boostings et au réseau pour améliorer le mélange à poids égaux, même s'ils sont plus faibles seuls.

## Protocole
v1 + one-hot des 14 _cat comme E30b (184 indicatrices, origine gardée, sauf pour la logistique où elle est remplacée).
Mêmes folds, test = moyenne des 5 modèles de fold, nombre d'itérations **fixe** (aucun early stopping), CPU n_jobs 8.
Mélange : chaque modèle est **ajouté comme 5e membre à poids égaux** à la composition de E33f (E30 + E07 + E08 + E32c),
soit un poids de 1/5. Je ne l'ai pas mélangé 50/50 avec E33f, ce qui lui aurait donné la moitié du poids.
Un seul mélange par modèle, pas de poids appris. Règle : 5/5 ou Δ > 2·sd(Δ)/√5.
Code : `usine/src/03_train.py` (configs E60-E63, `ajuster_logreg_riche`, `ajuster_extratrees`), `06_melange.py`. Jobs 60 à 66.

## Résultats
| Exp | Modèle | Gini CV | Corr. rangs OOF avec E30 / E07 / E08 / E32c | Temps |
|---|---|---|---|---|
| E60 | LightGBM DART (lr 0,05, 600 arbres, drop 0,1) | 0,27933 ± 0,0082 (OOF global 0,189 *) | 0,65 / 0,65 / 0,65 / 0,62 * | 8 min |
| E61 | LightGBM extra_trees=True (1400 arbres) | 0,27881 ± 0,0073 | 0,962 / 0,953 / 0,943 / 0,923 | 6 min |
| E62 | Logistique L2 (C 0,05), one-hot + RankGauss appris dans le fold | 0,25327 ± 0,0100 | 0,845 / 0,867 / 0,885 / 0,887 | 45 s |
| E63 | ExtraTreesClassifier (250 arbres, leaf 150, max_features 0,3) | 0,25686 ± 0,0062 | 0,863 / 0,869 / 0,863 / 0,841 | 14 min |

(*) Avec DART, l'échelle des probabilités dérive d'un fold à l'autre : le Gini de l'OOF global (0,189) et la corrélation
globale ne veulent rien dire. Le Gini par fold, lui, est correct. Conséquence pour le mélange : les rangs sont calculés sur
tout l'OOF, donc E60 y est un peu « compressé » dans certains folds (léger biais, sans incidence sur la conclusion).

| Mélange (rangs égaux) | Gini CV | Δ vs E33f (0,28531) | Folds | Verdict |
|---|---|---|---|---|
| E64 = E33f-membres + E62 (logistique) | 0,28300 ± 0,0080 | −0,00230 | 0/5 | **rejeté** |
| E65 = + E61 (extra_trees) | 0,28521 ± 0,0073 | −0,00009 | 2/5 | neutre |
| E66 = + E60 (DART) | 0,28504 ± 0,0075 | −0,00027 | 1/5 | **rejeté** |
| E67 = + E63 (ExtraTrees sklearn) | 0,28421 ± 0,0072 | −0,00110 | 0/5 | **rejeté** |

## Verdict
**Aucun modèle CPU n'améliore E33f.** Je n'ai donc pas testé d'union et **pas produit de 2_308.csv**.
- Les modèles peu corrélés (logistique, ExtraTrees, corrélation 0,84-0,89) sont trop faibles (−0,025 à −0,03 de Gini) :
  avec un poids égal de 1/5, ils dégradent le mélange. Le réseau DAE (0,275, corrélation ~0,91) reste le seul membre
  à la fois différent et assez fort.
- Les variantes de LightGBM (DART, extra_trees) sont presque aussi fortes que E30, mais redondantes avec E30, E07 et E08.
- Mélanges de la piste B dans le registre avant 14h15 : E56 (2_307, +0,00029 vs E33f, 4/5) est le meilleur. E65, le seul
  mélange à peu près neutre ici, ne justifie pas un mélange commun avec E56 : je ne l'ai pas testé, pour éviter la sélection.
- Temps : ~25 min de calcul (jobs en parallèle).

## Prochaine idée
Pour aller plus loin côté CPU, il faudrait des poids non égaux (un petit poids pour la logistique), choisis par validation
imbriquée (poids appris sur 4 folds, évalués sur le 5e). Le gain attendu est ≤ 0,0002 (cf. E17), ce qui ne vaut pas le risque avant le gel.
