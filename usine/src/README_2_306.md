# Soumission finale 2_306 — code et reproduction

**Modèle E33f** : moyenne des rangs, à poids égaux, de 4 modèles différents.
Gini CV 0,28531 ± 0,00738 (5 folds stratifiés, graine 42) ; Gini serveur 0,30079.
Fichier déposé : `usine/submissions/2_306.csv`.

## 1. Schéma de la chaîne

```
train.csv, test.csv
   │  01_audit.py           (copie typée -> usine/data/interim/*_raw.parquet)
   ▼
02_prepare.py  ──► données v1 = brut sans les 20 ps_calc_*  (usine/data/processed/v1/)
   │                                   + utils.py : folds figés usine/experiences/folds.csv
   ├──► 03_train.py --exp E30 ............ LightGBM (paramètres Jahrer, one-hot)  ─┐
   ├──► 03_train.py --exp E07 ............ XGBoost GPU                            ─┤
   ├──► 03_train.py --exp E08 ............ CatBoost GPU                           ─┤
   └──► 12_dae.py (DAE swap noise, sans cible, train+test)                         │
            └──► 13_nn_dae.py E32b / E32b_g1 / E32b_g2 (réseau sur activations)   │
                     └──► 06_melange.py E32c (moyenne des 3 graines) ─────────────┤
                                                                                  ▼
                                    06_melange.py E33f = rangs moyens (E30, E07, E08, E32c)
                                                                                  ▼
                              07_rangs_vers_proba.py -> usine/submissions/2_306.csv
```

Chaque modèle produit des prédictions **out-of-fold** (OOF, sur les mêmes 5 folds) et une prédiction test
(moyenne des 5 modèles de fold). Le mélange et ses scores sont calculés uniquement sur ces OOF.

## 2. Fichiers et rôles

| Fichier (`usine/src/`) | Rôle dans 2_306 |
|---|---|
| `utils.py` | Métrique Gini, folds figés (StratifiedKFold 5, graine 42), registre, écriture et contrôle de la soumission |
| `01_audit.py` | Audit des données + copie typée des CSV en parquet (entrée de 02) |
| `02_prepare.py` | Version de données **v1** (retrait des 20 `ps_calc_*`), contrôles de non-régression |
| `03_train.py` | Validation croisée des modèles d'arbres : configs **E30** (LightGBM), **E07** (XGBoost GPU), **E08** (CatBoost GPU) |
| `12_dae.py` | Autoencodeur débruiteur (swap noise 15 %, RankGauss + one-hot, 207-1500-1500-1500-207), non supervisé |
| `13_nn_dae.py` | Réseau 4500-1000-1000-1 sur les activations du DAE gelé, 6 époques fixes moyennées (E32b, 3 graines) |
| `06_melange.py` | Moyenne des rangs à poids égaux (E32c, puis E33f) + comparaison appariée + registre |
| `07_rangs_vers_proba.py` | Remet le classement de E33f à l'échelle des probabilités (quantile mapping) et écrit la soumission |

## 3. Commandes exactes, dans l'ordre (cwd = dossier `Hackathon`)

| # | Commande | Matériel | Durée |
|---|---|---|---|
| 1 | `python -u usine/src/01_audit.py` | CPU | ~2 min |
| 2 | `python -u usine/src/02_prepare.py` | CPU | < 1 min |
| 3 | `python -u usine/src/03_train.py --exp E07 E08 --base E02 --base-version v1` | GPU | ~7 min (E07 20 s, E08 6,5 min) |
| 4 | `python -u usine/src/03_train.py --exp E30` | CPU | ~2 min |
| 5 | `python -u usine/src/12_dae.py --budget-min 32 --batch 512` | GPU | 32 min (596 époques) |
| 6 | `python -u usine/src/13_nn_dae.py --exp E32b    --moyenne-epoques 6` | GPU | ~2-3 min |
| 7 | `python -u usine/src/13_nn_dae.py --exp E32b_g1 --moyenne-epoques 6 --graine 1` | GPU | ~2-3 min |
| 8 | `python -u usine/src/13_nn_dae.py --exp E32b_g2 --moyenne-epoques 6 --graine 2` | GPU | ~2-3 min |
| 9 | `python -u usine/src/06_melange.py --exp E32c --membres E32b E32b_g1 E32b_g2 --ref E32b` | CPU | < 1 min |
| 10 | `python -u usine/src/06_melange.py --exp E33f --membres E30 E07 E08 E32c --ref E31c` | CPU | < 1 min |
| 11 | `python -u usine/src/07_rangs_vers_proba.py --exp E33f --membres E30 E07 E08 E32b --sortie usine/submissions/2_306.csv` | CPU | secondes |

Total ≈ 50 min, dont 32 min de DAE.

Remarques importantes :
- **Étape 5 facultative** : les poids sont déjà dans `usine/data/dae/dae.pt` (copie identique `dae_E33f_596ep.pt`).
  Relancer l'étape 5 **écrase `dae.pt`** ; le DAE est entraîné à budget de temps et en TF32, donc non
  reproductible bit à bit. Pour reproduire 2_306 exactement : partir de l'étape 6 avec le `dae.pt` existant.
- Les fichiers OOF/test déjà présents dans `usine/experiences/oof/` suffisent pour rejouer les étapes 9 à 11
  à l'identique (vérifié par le reviewer, rapport 93 §1).
- Les options `--ref` ne servent qu'à la **comparaison appariée** (registre) : la référence doit déjà avoir ses
  fichiers OOF. Depuis un dossier vide, lancer d'abord les références :
  `03_train.py --exp E00 E01`, `03_train.py --exp E02`, `03_train.py --exp E13`, `03_train.py --exp E29`,
  `06_melange.py --exp E09 --membres E02 E07 E08 --ref E02` (après l'étape 3),
  `06_melange.py --exp E31c --membres E30 E07 E08 --ref E09` (après l'étape 4).
  Elles ne changent rien aux prédictions de 2_306.
- À l'étape 11, `--membres` ne fournit que les **valeurs** de probabilité ; l'**ordre** (donc le Gini) vient de E33f.
- Sur le PC du projet, les commandes passent par le lanceur `usine/runner.py` : une ligne d'arguments dans
  `usine/queue/<nom>.job` (préfixe `#gpu` pour les jobs GPU), log dans `usine/logs/<nom>.log`.

## 4. Où sont les sorties

| Sortie | Emplacement |
|---|---|
| Données préparées v1 | `usine/data/processed/v1/{train,test}.parquet`, `schema.json` |
| Folds figés | `usine/experiences/folds.csv` |
| OOF / test / Gini par fold de chaque expérience | `usine/experiences/oof/<EXP>_oof.csv`, `<EXP>_test.csv`, `<EXP>_folds_detail.csv` |
| Registre et comparaisons appariées | `usine/experiences/registre.csv`, `usine/experiences/comparaisons.csv` |
| DAE | `usine/data/dae/X_all.npy`, `meta.json`, `dae.pt` ; folds du réseau `usine/data/dae/<EXP>_fold<k>.npz` |
| Soumission | `usine/submissions/2_306.csv` (sans en-tête, `id,prediction`, ordre de `test.csv`) |
| Logs d'exécution | `usine/logs/` (36_dae, 39_nn_E32b, 42/43_nn_E32b_g*, 46_E32c, 47_E33f, 50_sub_2_306) |

## 5. Les 5 fichiers à montrer à l'oral (et les lignes clés)

1. **`utils.py`** — le protocole.
   - l. 50 `def gini` : Gini = 2·AUC − 1, ne dépend que de l'ordre.
   - l. 86 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` : 3,6 % de sinistres dans chaque fold ;
     folds écrits une fois dans `folds.csv` et relus partout.
2. **`03_train.py`** — les modèles d'arbres.
   - l. 143 `LGB_JAHRER` et l. 154 config **E30** : paramètres de la 1re place Kaggle, 1400 arbres fixes, one-hot.
   - l. 206 / 213 configs **E07** / **E08** (XGBoost / CatBoost GPU).
   - l. 436-438 : prédiction OOF du fold et moyenne des 5 modèles pour le test.
   - l. 369 `def comparer` : comparaison appariée fold par fold et règle de verdict.
3. **`12_dae.py`** — la représentation non supervisée.
   - l. 60 `rank_gauss` : chaque variable rendue gaussienne.
   - l. 127-134 `swap_noise` : 15 % des cases remplacées par la valeur d'un autre assuré.
   - l. 167 : perte = reconstruire la ligne propre à partir de la ligne bruitée.
4. **`13_nn_dae.py`** — le réseau supervisé.
   - l. 69 et l. 80 : DAE gelé, entrée = 3 × 1500 activations concaténées.
   - l. 86 `class Reseau` : 4500-1000-1000-1 avec dropout.
   - l. 127 et l. 140 : 6 époques fixes, moyenne des prédictions (aucun arrêt choisi sur le fold évalué).
5. **`06_melange.py`** + **`07_rangs_vers_proba.py`** — le mélange final.
   - `06_melange.py` l. 53 : passage en rangs ; l. 79 : poids égaux (non appris, score honnête).
   - `07_rangs_vers_proba.py` l. 46 : quantile mapping ; l. 53 : contrôle Spearman = 1 (Gini inchangé).

Chaque fichier commence par un en-tête (rôle dans 2_306, entrées, sorties, commande) et est découpé en
sections numérotées. Les commentaires ajoutés pour l'oral n'ont modifié aucune ligne de code
(vérification automatique : AST identique avant/après, sauvegardes `*.avant_commentaires.bak`).
