# 12 — ML engineer, piste B : autoencodeur débruiteur (DAE) + réseau de neurones (E32, E33)

**Source** : solution gagnante Kaggle *Porto Seguro's Safe Driver Prediction* (Michael Jahrer, 1re place, 2017,
post « 1st place with representation learning » ; recette transmise par l'utilisateur sous forme de capture).
**Code** : `usine/src/12_dae.py` (préparation + DAE), `usine/src/13_nn_dae.py` (réseau, E32*),
`usine/src/14_comparaisons_pisteB.py`, `usine/src/15_verif_test_E32.py` ; mélanges via `06_melange.py`.
**Matériel** : RTX 4060 Laptop 8 Go, PyTorch 2.14.1+cu126 (TF32 activé), 12 threads CPU max.

## 1. Hypothèse
Les 3 boostings sont corrélés à 0,96–0,99 : le mélange plafonne. Un réseau appris sur une représentation
non supervisée (DAE à bruit par échange) classe les assurés différemment et doit apporter de la diversité.

## 2. Ce qui a été fait (et réduit par rapport à l'original)
| Élément | Jahrer (original) | Ici | Pourquoi |
|---|---|---|---|
| Entrées | calc retirées, one-hot _cat, RankGauss | idem : **207 colonnes** (12 RankGauss, 11 binaires, one-hot des 14 _cat, -1 = modalité) | notre jeu (446 409 + 148 803 lignes) a moins de modalités → 207 au lieu de 221 |
| DAE | 221-1500-1500-1500-221, swap 15 %, MSE | **même architecture**, swap 15 %, MSE, 5,1 M paramètres | — |
| Données du DAE | train + test | train + test (595 212 lignes, **sans cible** : légitime, mais signalé) | — |
| Optimisation DAE | batch 128, lr 3e-3 × 0,995/époque, 1000 époques (~5 h) | Adam, **batch 512**, lr 1e-3 décroissant selon le temps (× 0,05 en fin de budget), **596 époques en 32 min** (3,1 s/époque) | budget ≤ 40 min |
| Réseau | 4500-1000-1000-1, dropout 0,5 / entrée 0,1, L2 0,05, lr 1e-4 | idem sauf **L2 = weight_decay 1e-5** (Adam), batch 256 | L2 0,05 de son framework non transposable tel quel ; pas eu besoin de régler |
| Sélection | — | E32 : early stopping par fold ; **E32b : 6 époques fixes, moyenne des prédictions des 6 époques** | supprimer le biais de sélection sur le fold évalué |
| Graines | — | E32c = moyenne de rangs de 3 graines de E32b | réduire la variance |

Activations calculées à la volée par lots sur le GPU (595 k × 4500 en float32 = 10,7 Go impossibles à stocker).
Le réseau est rapide : **~2 min pour 5 folds** (3,6 s/époque). DAE : reconstruction propre MSE 0,0031 → 0,00042, bruitée 0,0112 → 0,0059 (log `usine/logs/36_dae.log`, poids `usine/data/dae/dae.pt`).

## 3. Résultats (folds figés `folds.csv`, Gini par fold)
| Exp | Description | Gini CV | Spearman OOF vs E09 | Comparaison appariée | Verdict |
|---|---|---|---|---|---|
| E32 | réseau sur DAE, early stopping par fold (époques 2,3,5,1,5) | 0,27114 ± 0,00802 | 0,874 | vs E09 −0,0123 (0/5) | modèle seul, pour mélange |
| E32b | idem, **6 époques fixes, moyenne des époques** | **0,27413 ± 0,00850** | 0,913 | vs E32 **+0,0030 (5/5)** | garde (meilleur et sans biais) |
| E32c | E32b × 3 graines (rangs) | 0,27517 ± 0,00785 | 0,924 | vs E32b +0,0010 (4/5) | neutre |
| E33 | rangs égaux E02+E07+E08+E32 | 0,28493 ± 0,00779 | — | vs E09 +0,00150 (5/5) | garde |
| E33b | rangs 50/50 E09 + E32 | 0,28340 ± 0,00778 | — | vs E09 −0,00003 (3/5) | neutre (trop de poids au réseau) |
| E33d | rangs égaux E02+E07+E08+E32b | 0,28494 ± 0,00796 | — | vs E09 +0,00152 (5/5) | garde |
| E33c | E30+E07+E08+E32 (avec LightGBM piste A) | 0,28548 ± 0,00744 | — | vs E31c +0,00123 (4/5) | garde |
| E33e | E30+E07+E08+E32b | 0,28536 ± 0,00758 | — | vs E31c +0,00111 (5/5) ; vs E09 +0,00194 (5/5) | garde |
| **E33f** | **E30+E07+E08+E32c** | **0,28531 ± 0,00738** | — | **vs E31c +0,00105 (5/5) ; vs E09 +0,00188 (5/5, seuil 2se 0,00036)** | **garde → 2_306** |

Le réseau seul est ~0,008 sous les boostings, mais c'est le **plus gros gain de mélange de la journée** (+0,0015 à +0,0019 vs E09, 5/5),
parce qu'il classe différemment (Spearman 0,87–0,92 contre 0,96–0,99 entre boostings). Poids égaux à 1/4 : le 50/50 (E33b) n'apporte rien.

**Biais et réserves (honnêtes)**
- E32 (early stopping sur le fold évalué) est optimiste : à époque commune, −0,0022. E32b supprime ce biais ; il reste un biais faible :
  le choix « 6 époques » a été fait en regardant les courbes de E32 (toutes les meilleures époques entre 1 et 5).
- Le DAE a vu le test (sans cible) ; c'est la pratique de l'original et c'est légitime, mais le jury doit le savoir.
- E33f vs E33e : −0,00006 (neutre) ; je préfère E33f (3 graines) pour la stabilité : Δ par fold vs E09 plus homogènes (0,0014–0,0025).
- Contrôle test : Spearman E32b/E09 = 0,913 en OOF, 0,932 sur le test (cohérent : test = moyenne de 5 réseaux, plus lisse) ;
  moyenne des probas du réseau 0,0378 (OOF) / 0,0379 (test).

## 4. Soumission préparée (non déclarée déposable)
`usine/submissions/2_306.csv` = E33f remis à l'échelle proba (`07_rangs_vers_proba.py`, valeurs = moyenne des probas E30, E07, E08, E32b ;
Spearman sortie/mélange 1,000000). Auto-contrôle : 148 803 lignes, sans en-tête, ids dans l'ordre de test.csv, 0 NaN, LF,
moyenne 0,0367 (taux train 0,0364), min 0,0075, max 0,3946. Spearman vs 2_304 : 0,9935.

## 5. Temps de calcul
Installation/contrôle torch ~2 min ; préparation + DAE 33 min ; E32, E32b, E32b_g1, E32b_g2 : ~2 min chacun ; mélanges < 1 min.

## 6. Prochaine idée recommandée
Le DAE est le facteur limitant (596 époques vs 1000 à batch 128 dans l'original, soit ~5× moins de pas de gradient) :
si du temps GPU reste avant le gel, relancer le DAE ~40 min de plus en repartant des poids actuels, puis refaire E32b/E33f (≈5 min).
Second levier : 2–3 graines de plus du réseau (coût 2 min chacune), effet attendu faible.

## 7. Reprise du DAE (feu vert lead-tech, 12h08–12h55 Paris)
**Hypothèse** : le DAE était sous-entraîné (596 époques) ; le prolonger devait améliorer la représentation et donc le réseau.
**Changement unique** : DAE repris depuis `dae.pt` pendant 38 min de plus (724 époques, **1320 au total**), Adam lr 3e-4 → 3e-5
(décroissance selon le temps), même bruit par échange (15 %) et même batch (512). Sauvegardé sous `usine/data/dae/dae_reprise.pt` ;
`dae.pt` est intact (copie supplémentaire `dae_E33f_596ep.pt`), donc E33f reste reproductible. Réseau et protocole inchangés
(E32b : 6 époques fixes, moyenne des prédictions des époques ; graines 0/1/2) ; code : options `--reprise/--sortie` (12_dae.py)
et `--dae` (13_nn_dae.py), comparaisons `usine/src/16_comparaisons_reprise.py`.

Reconstruction : MSE propre 0,00042 → 0,00039, bruitée ~0,0059 → ~0,0058 : **quasi plateau**, le DAE avait déjà presque convergé.

| Exp | Description | Gini CV | Comparaison appariée | Verdict |
|---|---|---|---|---|
| E40 | E32b avec dae_reprise.pt (graine 0) | 0,27328 ± 0,00816 | vs E32b −0,00085 (1/5) | rejete |
| E40c | E40 × 3 graines (rangs) | 0,27450 ± 0,00745 | vs E32c −0,00066 (1/5) | rejete |
| E41 | rangs égaux E30 + E07 + E08 + E40c | 0,28511 ± 0,00726 | **vs E33f −0,00019 (1/5, seuil 2se 0,00018)** ; vs E09 +0,00169 (5/5) | **rejete vs E33f** |

**Verdict** : prolonger le DAE n'apporte rien (léger recul, à la limite du bruit). **E33f reste le meilleur mélange ; pas de 2_307.**
Lecture : à cette capacité (3 × 1500) et avec notre volume de données, la représentation sature vers ~600 époques à batch 512 ;
le nombre d'époques de l'original (1000 à batch 128) n'est pas le facteur limitant ici. Spearman OOF E40 vs E09 0,913 (identique à E32b).
Temps : DAE 38 min, 3 réseaux × ~2 min, mélanges < 1 min.
**Prochaine idée** (non recommandée avant le gel) : plus de diversité plutôt que plus d'époques, par ex. un 2e DAE de graine différente
ou un swap noise différent (10 % / 20 %), à mélanger avec E32c ; gain attendu faible (≤ 0,0005).
