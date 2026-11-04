# 93 - Reviewer : revue finale avant gel (2_306 vs 2_307)

Date : 2026-11-04 ~14h07 Paris. Script de contrôle : `usine/src/93_revue_finale.py` (numpy + pandas, n'écrit ni registre ni OOF, 13 s CPU).
Commande : `python -u usine/src/93_revue_finale.py` (cwd = Hackathon).

## 1. Format (mon script, lecture brute des octets)

| Fichier | Lignes | En-tête | CR | LF final | 2 colonnes | ids = test.csv, même ordre | doublons | NaN | min | max | moyenne |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2_306.csv | 148 803 | non (1re ligne `7,0.04507…`) | non | oui | oui | oui | 0 | 0 | 0,00748 | 0,39459 | 0,03674 |
| 2_307.csv | 148 803 | non (1re ligne `7,0.04508…`) | non | oui | oui | oui | 0 | 0 | 0,00747 | 0,38154 | 0,03694 |

- Ordre de 2_307 = E56_test : Spearman 1,000000 ; recalcul du quantile mapping (membres proba E30 E07 E08 E32b E51_g0) : écart max 1,06e-3,
  dû uniquement aux 10 142 ex-aequo du mélange de rangs (départagés par ordre d'id, déjà signalé revue 92, sans effet).
- E51c, E56r, E56 recalculés depuis leurs membres : écart max 6,7e-6 (OOF et test). Ids, folds, cible identiques dans tous les OOF.
- Taux moyen 0,0369 vs taux train ~0,0365 : plausible.

## 2. Fuites (E50 / E51c / E56)

- E50 (`12_dae.py --sortie dae_swap07.pt --swap 0.07`) : non supervisé, la cible est retirée avant RankGauss (`12_dae.py` l.63
  `train.drop(columns=[U.TARGET])`). Appris sur train + test sans cible : même statut que le DAE de E32 (transductif, à mentionner au jury).
- E51_g0/g1/g2 (`13_nn_dae.py --dae dae_swap07.pt --moyenne-epoques 6 --graine k`) : folds figés (`folds.csv`), un réseau par fold,
  6 époques fixes moyennées (folds_detail : 6 partout) -> aucune décision prise sur le fold de validation. DAE gelé.
- E56 : mélange de rangs à poids égaux (poids OOF-optimaux calculés mais non retenus). Pas de fuite détectée.

## 3. Le gain E56 vs E33f est-il réel ?

Gini par fold, apparié vs E33f :

| Mélange | Δ/fold | Δ moyen | se | gagnés |
|---|---|---|---|---|
| **E56** | +0,00055 +0,00026 +0,00021 −0,00002 +0,00043 | **+0,00029** | 0,00010 | 4/5 |
| E59 (E51c remplace E32c) | +0,00083 +0,00028 +0,00015 −0,00035 +0,00061 | +0,00030 | 0,00020 | 4/5 |
| E55 (5 membres) | | +0,00015 | 0,00010 | 3/5 |
| E58 / E54 / E57 / E53 (avec E52) | | −0,0003 à −0,0015 | | 0-1/5 |

- **Par graine** (E56 recalculé avec une seule graine réseau g, comparé à E33f avec la même graine g) : +0,00036 (4/5), +0,00031 (5/5),
  +0,00023 (5/5). Le gain ne dépend pas d'une graine favorable : il tient sur les 3 réentraînements du réseau.
- **Bootstrap des lignes dans les folds** (50 tirages) : E56 − E33f = +0,00030, IC90 [+0,00005 ; +0,00053], P(Δ ≤ 0) = 4 %.
  Bootstrap des 5 folds : P(Δ moyen ≤ 0) = 0 % (le seul fold perdant est à −0,00002).
- **Sélection** : 7 variantes testées (E53-E59) ; les 4 avec E52 sont nettement négatives (idée réfutée, pas une recherche fine).
  Le vrai tri se fait entre E55/E56/E59, très corrélées. Dans le bootstrap, le « meilleur » parmi {E33f, E55, E56, E59} est
  E59 28/50, E56 12/50, E55 8/50, **E33f 2/50** : l'ordre entre variantes à réseau E51 est du bruit, mais **l'apport de E51c, lui, est stable**.
  Malédiction du vainqueur estimée ~+0,0001 : gain attendu hors échantillon ≈ +0,0001 à +0,0002, **probabilité de perte faible**.
- **Robustesse** : sd entre folds E56 0,00732 vs E33f 0,00738 ; E56 met moins de poids sur chaque réseau (2 réseaux à 1/8) que E59,
  préférable à E59 malgré un score égal (Δ/fold E59 plus dispersés).
- **Proximité test** : Spearman 2_306 vs 2_307 = **0,99950** ; écart de rang moyen 0,66 percentile ; recouvrement du top 3,7 % : 97,3 %.
  Les deux fichiers sont quasi identiques : l'écart officiel attendu est de l'ordre de 1e-4 dans un sens ou l'autre.

Non vérifié par moi : autre découpage de folds (réentraîner 6 réseaux + arbres ne tient pas dans le délai). Couvert en partie par les 3 graines.

## 4. Recommandation

**Déposer `2_307.csv` en dernier.** Gain petit (+0,0003) mais cohérent sur 4/5 folds, 3/3 graines, bootstrap 96 %, sans fuite,
format conforme. Risque de régression négligeable (fichier à 0,9995 de 2_306). 2_306 reste le repli valide.

**Chiffre à annoncer à l'oral** : « Gini CV 0,2856 ± 0,007 (5 folds), attendu ≈ 0,285 hors échantillon ; +0,0021 vs notre LightGBM
de référence E09, +0,0003 apporté par le 2ᵉ autoencodeur ». Ne pas présenter le +0,0003 comme significatif au sens fort :
c'est un affinement. Mentionner l'apprentissage non supervisé des DAE sur train + test sans la cible.

## 5. Reproductibilité

Poids présents : `data/dae/dae.pt` (= `dae_E33f_596ep.pt`, md5 identiques 92f35f49…), `data/dae/dae_swap07.pt`, `data/dae/X_all.npy`, `meta.json`.
Chaîne (cwd = Hackathon ; E30, E07, E08 : commandes inchangées, voir registre / rapport 92) :

```
# 2_306
python -u usine/src/13_nn_dae.py --exp E32b    --moyenne-epoques 6              # GPU, relit dae.pt
python -u usine/src/13_nn_dae.py --exp E32b_g1 --moyenne-epoques 6 --graine 1
python -u usine/src/13_nn_dae.py --exp E32b_g2 --moyenne-epoques 6 --graine 2
python -u usine/src/06_melange.py --exp E32c --membres E32b E32b_g1 E32b_g2 --ref E32b
python -u usine/src/06_melange.py --exp E33f --membres E30 E07 E08 E32c --ref E31c
python -u usine/src/07_rangs_vers_proba.py --exp E33f --membres E30 E07 E08 E32b --sortie usine/submissions/2_306.csv
# 2_307 (en plus)
python -u usine/src/12_dae.py --sortie dae_swap07.pt --swap 0.07 --budget-min 28 --batch 512 --lr 1e-3   # facultatif : poids déjà présents
python -u usine/src/13_nn_dae.py --exp E51_g0 --dae dae_swap07.pt --moyenne-epoques 6 --graine 0          # idem g1, g2
python -u usine/src/06_melange.py --exp E51c --membres E51_g0 E51_g1 E51_g2 --ref E51_g0
python -u usine/src/06_melange.py --exp E56r --membres E32c E51c --ref E32c
python -u usine/src/06_melange.py --exp E56  --membres E30 E07 E08 E56r --ref E33f
python -u usine/src/07_rangs_vers_proba.py --exp E56 --membres E30 E07 E08 E32b E51_g0 --sortie usine/submissions/2_307.csv
```

Le 12_dae.py est à budget de temps (nombre d'époques variable) et le réseau tourne sur GPU en TF32 : réentraînement non bit à bit,
mais les poids sauvegardés et les fichiers OOF/test permettent de régénérer exactement les mélanges et les soumissions (vérifié §1).

## Constats

**BLOQUANT** : aucun.

**IMPORTANT**
1. Gain E56 faible et partiellement sélectionné (7 variantes, E59 aussi bon) : à annoncer comme affinement (+0,0003), pas comme progrès net.
   Preuve : §3, bootstrap meilleur = E59 28/50. Correction : formulation orale ci-dessus ; ne plus tester de variante avant le gel.

**MINEUR**
2. Quantile mapping de 2_307 sur E51_g0 seule (pas E51c) : sans effet sur le Gini (ordre = E56, Spearman 1,000000).
3. 10 142 ex-aequo dans E56_test, départagés par id : effet négligeable.
4. Fichier `submissions/_check_2_305_regen.csv` présent dans le dossier des soumissions : ne pas le déposer par erreur.
5. Autre découpage de folds non testé (délai).

## Verdict

**VALIDÉ AVEC RÉSERVES** (réserve = taille du gain, aucune sur la validité des fichiers).
- `2_307.csv` : **DÉPOSABLE, à déposer en dernier**.
- `2_306.csv` : DÉPOSABLE (repli).
- Oral : Gini CV 0,2856 ± 0,007, attendu ≈ 0,285.
