# 08 — ML engineer : moyenne multi-graines (E17), poids non égaux (E18), corrections du reviewer

Heure : 10h15–10h50 (Paris). Données v1, folds communs `usine/experiences/folds.csv` (StratifiedKFold 5, graine 42), métrique Gini (pas de seuil).

## 1. Corrections demandées par le reviewer (revue 91)
- **Règle de verdict durcie** (`03_train.comparer`) : `garde` si Δmoy > 0 et (5/5 folds gagnés **ou** Δmoy > 2·sd(Δ)/√5) ; `rejete` symétrique (0/5 ou −Δmoy > 2·sd/√5) ; sinon `neutre`. sd(Δ) avec ddof=1 (erreur-type, équivalent d'un test t à ~2 σ). La colonne `d_std` de `comparaisons.csv` reste l'écart-type ddof=0 (inchangée).
- **Registre** : `ajouter_registre` écrit en LF (`lineterminator="\n"`) ; les 17 lignes CRLF existantes ont été normalisées (0 CR). Sauvegarde : `experiences/registre_avantE17.csv.bak`.
- **`delta_vs_meilleur` renommée `delta_vs_ref`** = Δ apparié moyen par fold vs la référence de l'expérience (nommée dans le commentaire). Valeurs historiques recalculées depuis `comparaisons.csv` (n/a pour E00 et E01). Le registre ne calcule plus de verdict « gain > std » : le verdict vient toujours de la comparaison appariée.
- Re-lecture de l'historique avec la nouvelle règle (verdicts du registre NON réécrits, pour la traçabilité) : **E02 garde → neutre** (Δ +0,0022, 4/5, seuil 0,0031), **E11 et E12 rejete → neutre**. E09 et E13 restent `garde` (5/5). Rien d'autre ne change.
- Sauvegardes du code avant modification : `src/03_train_v2_avantE17.py.bak`, `src/utils_avantE17.py.bak`.

## 2. E17 — moyenne multi-graines
**Hypothèse** : moyenner 3 graines de modèle par membre réduit la variance → petit gain, surtout plus de stabilité sur le test.
**Changement** : graines de modèle 42 (= expérience d'origine, même code et mêmes paramètres), 7 et 2026, **mêmes folds** ; probabilités moyennées par membre ; puis mélange des rangs à poids égaux.
Code : `03_train.py --graines 7 2026` (variantes `<EXP>s7`, `<EXP>s2026`, hors registre), `src/08_multigraines.py` (moyenne par membre, diagnostic).

| Exp | Contenu | Gini CV | Δ apparié | Folds | Verdict |
|---|---|---|---|---|---|
| E13s7 / E13s2026 | LightGBM E13, graine 7 / 2026 | 0,28180 / 0,28156 | +0,00001 / −0,00023 vs E13 | — | ingrédients |
| **E17a** | E13 moyenné sur 3 graines | 0,28217 ± 0,00782 | +0,00038 vs E13 (seuil 0,00057) | 3/5 | neutre |
| **E17** | rangs égaux E17a + E07 + E08 | **0,28371 ± 0,00766** (OOF 0,28358) | +0,00003 vs E14 (seule différence : LightGBM 3 graines) | 2/5 | neutre |
| E17 vs E09 | (2 changements : lr 0,01 + 3 graines) | | +0,00028 (sd 0,00033, seuil 0,00033) | 4/5 | neutre (positif) |

- Écart entre graines LightGBM : 0,00024 en Gini CV ; corrélation OOF entre graines 0,993–0,994. Le bruit de graine est donc **petit devant le bruit de découpage** (std entre folds 0,0077 ; reviewer : 0,276–0,279 selon la graine des folds).
- Dans le mélange, la moyenne des graines LightGBM n'apporte rien de mesurable (+0,00003) : le mélange de 3 modèles différents moyenne déjà la variance.
- **XGBoost et CatBoost multi-graines NON obtenus — INCIDENT GPU** : le job `13_graines_gpu` (E07s7, E07s2026, E08s7, E08s2026), lancé à 10h17, n'a pas terminé un seul fold XGBoost (normalement 4 s/fold). **Aucun job GPU n'a progressé depuis** : dernier fold de `91_revue_E09_s7` écrit à 10h16, de `91_revue_E09_s42` à 10h07 (≈ 5–9 min/fold auparavant). Trois processus GPU simultanés (2 CatBoost qui réservent par défaut ~95 % de la mémoire GPU + XGBoost) semblent saturer/bloquer le GPU. Le lanceur n'a pas de commande d'annulation : il faut **tuer à la main sur le PC le processus python de `13_graines_gpu`** (gestionnaire des tâches, ligne de commande contenant `E07s7`) pour libérer le GPU du reviewer. S'il est relancé plus tard (GPU libre, seul), `08_multigraines.py membres --quels E17b E17c` puis `06_melange.py --exp E19 --membres E17a E17b E17c --ref E17` donnent la version complète (E19). Vu E17a, gain attendu ≈ 0.

## 3. E18 — poids non égaux : PASSÉ (non justifié)
Test sans biais : poids choisis (grille pas 0,05) sur 4 folds, évalués sur le 5e.
- Membres E13/E07/E08 : Δ poids appris − poids égaux = +0,00006 (sd 0,00056, 3/5).
- Membres E17a/E07/E08 : +0,00011 (sd 0,00054, 3/5).
Les poids appris vont vers ~0,45 LightGBM / 0–0,2 XGBoost / 0,45 CatBoost (XGBoost très corrélé à LightGBM : 0,991), mais le gain hors échantillon est nul. Le 0,5/0,25/0,25 proposé n'est pas soutenu par les OOF. Poids égaux conservés.

## 4. Soumission candidate (NON déclarée déposable)
`usine/submissions/2_304.csv` = E17 remis à l'échelle probabilité par `07_rangs_vers_proba.py --exp E17 --membres E17a E07 E08` (job `17_sub_2_304`).
Auto-contrôle : 148 803 lignes, sans en-tête, ids = ordre de test.csv, 0 NaN, 0 CR, valeurs dans [0,008 ; 0,466], moyenne 0,0364 (taux de positifs train 0,0364), Spearman sortie/mélange = 1,000000, Spearman vs 2_303 = 0,9998.
Intérêt vs 2_303 : quasi identique (Δ CV +0,0003, non significatif) ; seulement un peu moins dépendante d'une graine LightGBM. À déposer seulement si le lead veut utiliser un dépôt ; sinon 2_303 reste équivalente.

## 5. Temps de calcul
E13s7 109,6 s, E13s2026 115,2 s (LightGBM 16 threads, en parallèle du job GPU) ; E17a, E17, diagnostics : < 1 min chacun ; job GPU : bloqué (aucun fold en 20 min, à tuer).

## 6. Prochaine idée recommandée
La variance de modèle est épuisée (graines, poids, réglages : tous < 0,0005). Le seul levier encore crédible est la **diversité** : ajouter un membre réellement différent (ex. réseau/logistique sur encodages cibles en folds, ou LightGBM sur v4 du data cleaner si ses variables apportent du signal), à juger par Δ apparié vs E09/E17. Sinon, figer E09/E17 et consacrer le temps restant à la soutenance.
