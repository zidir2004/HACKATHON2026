# 92 - Reviewer : revue des pistes A (LightGBM Jahrer) et B (DAE + réseau), soumissions 2_305 et 2_306

Revue du 2026-11-04, ~12h05-12h20 (Paris). Script de contrôle (lecture seule, n'écrit ni registre ni OOF) :
`usine/src/92_revue_pistesAB.py` (numpy + pandas, 15 s CPU, lancé dans le shell Linux).

## Verdicts

| Fichier | Contenu | Verdict |
|---|---|---|
| `usine/submissions/2_305.csv` | E34 = rangs égaux E32A (E30b, 3 graines) + E07 + E08 | **DÉPOSABLE** |
| `usine/submissions/2_306.csv` | E33f = rangs égaux E30 + E07 + E08 + E32c (réseau sur DAE, 3 graines) | **DÉPOSABLE, à privilégier** |

**Chiffre à annoncer (2_306)** : Gini CV 5 folds **0,285 ± 0,007** (écart-type entre folds ; moyenne 0,28531, OOF global 0,28517).
Estimation honnête du score attendu hors échantillon : **≈ 0,284** (correction de sélection −0,0005 à −0,001, voir §5),
fourchette plausible 0,277-0,292 (dispersion entre folds). Gain vs E34 : +0,001 ; vs E09 : +0,0019 (tous deux 5/5).

## 1. Format (contrôlé par mon script, lecture brute des octets)

| Contrôle | 2_305 | 2_306 |
|---|---|---|
| Lignes | 148 803 | 148 803 |
| En-tête | aucune (1re ligne `7,0.04593276`) | aucune (`7,0.04507021`) |
| ids = test.csv, même ordre, sans doublon | oui | oui |
| Séparateur `,`, 2 colonnes partout ; LF, pas de CR | oui | oui |
| NaN | 0 | 0 |
| Plage | [0,00749 ; 0,41956] | [0,00748 ; 0,39459] |
| Moyenne (taux train 0,03645) | 0,03636 | 0,03674 |
| Spearman vs mélange de rangs source | 1,000000 vs `E34_test.csv` | 1,000000 vs `E33f_test.csv` |
| Valeurs recalculées par mon script | écart ≤ 5e-9 hors ex-aequo | écart ≤ 5e-9 hors ex-aequo |

Spearman 2_305 vs 2_306 = 0,9958.

## 2. Incident de collision E32/E33 : aucun fichier écrasé n'a servi

Horodatages (shell en UTC, Paris = +1 h) : `E34_oof/test` 11h21:58, `2_305.csv` **11h24:51**, écrasement de `E32_*` par le réseau **11h50:00**.
Preuves indépendantes de l'heure :
- J'ai **reconstruit E32A** = moyenne des probas de E30b, E30bs7, E30bs2026 (méthode de `09_verif_graines.py`) puis E34 = rangs égaux
  E32A + E07 + E08 : identique à `E34_oof.csv` / `E34_test.csv` à 3e-16 près.
- Remise à l'échelle de 2_305 recalculée avec E32A : écart ≤ 5e-9 hors ex-aequo (moyenne 0,03636). Avec le `E32_test.csv` actuel (réseau) :
  écart 0,031 et moyenne 0,03711 : **2_305 a bien été produit avec le E32 de la piste A**.
- E32c (E32b + g1 + g2) et E33f (E30 + E07 + E08 + E32c) recalculés : identiques aux fichiers (≤ 3e-6, bruit d'ex-aequo).
  E33f n'utilise ni E32 ni E33 : aucune exposition à la collision. 2_306 reproduit avec E30, E07, E08, E32b comme documenté.
- Tous les OOF utilisés ont les mêmes ids, cibles et colonnes `fold` que E09 (folds figés `folds.csv`).

## 3. Fuites, piste B

- **DAE sur train + test sans cible : acceptable.** `12_dae.py` n'utilise que `train.drop(columns=[TARGET])` et test ; perte MSE de reconstruction.
  C'est transductif mais non supervisé, comme dans la solution d'origine. À signaler au jury, pas à corriger.
- **RankGauss et one-hot sans cible** : calculés sur train ∪ test (`preparer_entrees`), aucune référence à la cible. OK.
- **Réseau par fold** : `un_fold` entraîne sur `i_tr = fold != k` uniquement ; le fold k n'est que prédit. En mode `--moyenne-epoques`
  aucune décision n'est prise sur la validation (le Gini de validation est seulement affiché). Test = moyenne des 5 réseaux
  (`np.mean(p_te, axis=0)`), chacun moyenné sur ses 6 époques. OK.
- **Le DAE a vu (sans cible) les lignes de validation de chaque fold : pas d'optimisme de la CV.** La situation est symétrique
  avec le dépôt : les lignes du test ont elles aussi été vues par le DAE sans cible. La CV reproduit donc exactement les conditions
  du test (représentation apprise sur des lignes dont on prédira la cible, sans cette cible). L'optimisme n'apparaîtrait que si le
  DAE avait vu la cible ou n'avait pas vu le test (alors la CV serait plus favorable que le test) ; aucun des deux cas ici.
- **Choix « 6 époques » après lecture des courbes de E32 : biais ≈ 0.** Dans `logs/39_nn_E32b.log`, le Gini de la moyenne des époques
  1..k, moyenné sur les 5 folds, vaut 0,27310 (k=4), 0,27361 (k=5), 0,27414 (k=6) : il **monte encore** à 6. Le choix n'est pas
  un maximum de la courbe évaluée, et la quantité choisie (un entier, réglée sur l'early stopping d'une autre variante) n'a
  pas pu être ajustée au bruit des folds. Biais estimé < 0,0003 sur le réseau seul, < 0,0001 dans le mélange (poids 1/4).
- Hyperparamètres du réseau (lr 1e-4, wd 1e-5, batch 256) fixés a priori ; `T32` n'était qu'un essai de fonctionnement (2 époques, fold 0).

Piste A : one-hot sur train ∪ test sans cible (`03_train.one_hot`), 1400 arbres fixes, pas d'early stopping. E34 n'utilise pas E35 (1050 arbres, choisi après diagnostic). OK.

## 4. Réalité du gain (recalculé depuis les OOF)

| Exp | Gini par fold | Moyenne ± sd |
|---|---|---|
| E09 | 0,28079 / 0,27007 / 0,29180 / 0,28847 / 0,28600 | 0,28343 ± 0,00758 |
| E34 | 0,28121 / 0,27089 / 0,29250 / 0,28925 / 0,28745 | 0,28426 ± 0,00763 |
| E33f | 0,28220 / 0,27259 / 0,29365 / 0,29034 / 0,28775 | 0,28531 ± 0,00738 |

| Comparaison appariée | Δ par fold | Δ moyen | erreur-type | Folds |
|---|---|---|---|---|
| E33f − E09 | +0,00141 +0,00253 +0,00185 +0,00187 +0,00175 | **+0,00188** | 0,00018 | 5/5 |
| E34 − E09 | +0,00042 +0,00082 +0,00070 +0,00079 +0,00146 | +0,00084 | 0,00017 | 5/5 |
| **E33f − E34** | +0,00099 +0,00171 +0,00116 +0,00109 +0,00030 | **+0,00105** | 0,00023 | **5/5** |
| E33f − E33e | +0,00046 +0,00006 −0,00004 −0,00092 +0,00016 | −0,00006 | 0,00023 | 3/5 (neutre) |

Bootstrap des lignes à l'intérieur des folds (60 tirages, folds et modèles fixes) : E33f − E34 = +0,00098, IC90 [+0,00039 ; +0,00155],
P(Δ ≤ 0) = 0/60 ; E33f − E09 = +0,00184, IC90 [+0,00104 ; +0,00263]. **Le gain de E33f sur E34 est réel à l'échelle de l'échantillon.**
Limite : le bootstrap ne couvre pas la variance d'entraînement (autre découpage) ; non refait (GPU occupé, réseau non relançable sur CPU dans le délai).
Les 3 graines du réseau (E32b, g1, g2) donnent toutes un mélange gagnant (E33e, E33f), ce qui couvre en partie ce risque.

**Plausibilité du gain avec un membre plus faible (0,275)** : oui. Corrélations de rangs OOF : réseau vs arbres 0,90-0,92,
arbres entre eux 0,94-0,99 (E30/E32A 0,990). Le réseau apporte une erreur peu corrélée ; avec un poids 1/4 son déficit
de 0,009 est compensé. C'est le mécanisme décrit par Jahrer (son gain venait des réseaux), et les mélanges sans arbres
nouveaux (E33d, E33g : E02+E07+E08+réseau) gagnent aussi 5/5 vs E09 : l'effet ne dépend pas d'un membre particulier.

**Membres de E33f** : E30 (1 graine, _cat d'origine retirées) plutôt que E32A (3 graines) : cohérent avec E31c ≥ E31
(0,28425 vs 0,28418, écart non significatif). Ce choix n'est pas optimal de façon démontrée mais il est sans conséquence ;
**ne pas tester d'autres variantes maintenant** (ajouterait de la sélection pour un gain < 0,0002).
Échelle proba de 2_306 : quantile mapping sur la moyenne des probas E30, E07, E08, E32b ; moyenne 0,0367 (+0,7 % relatif vs taux
train, dû au réseau un peu haut, 0,0379). Sans effet sur le Gini ; acceptable.

## 5. Biais de sélection (score à annoncer)

Une quinzaine de mélanges ont été évalués sur les mêmes 5 folds aujourd'hui (E09, E17, E24, E31/b/c, E33/b-g, E34, E37) et des
dizaines de modèles. Le choix final se fait entre candidats séparés de 0,0001-0,0005 avec une erreur-type de ~0,0002 :
malédiction du vainqueur estimée à **+0,0002 à +0,0004** sur le dernier choix, environ **+0,0005 à +0,001** cumulés sur la journée.
L'idée « ajouter un réseau DAE » a été décidée a priori (recette Jahrer), pas trouvée par recherche, ce qui limite le biais sur le
gain principal. Annonce : **CV 0,285 ± 0,007, attendu ≈ 0,284 hors échantillon** ; le classement réel peut s'écarter de ±0,007.

## Constats

**BLOQUANT** : aucun.

**IMPORTANT**
1. **Reproductibilité de 2_306 menacée par la reprise du DAE.** `12_dae.py` sauve toutes les 10 époques dans `usine/data/dae/dae.pt`
   (même chemin) ; `13_nn_dae.py` relit `dae.pt`. La reprise de l'entraînement va écraser les poids (596 époques, 10h47 UTC)
   qui ont produit E32b/E32c/E33f/2_306. Correction : **copier `dae.pt` en `dae_E32_596ep.pt` avant la reprise** (ou sauver la reprise
   sous un autre nom) et nommer les nouvelles expériences avec de nouveaux identifiants (E38+), jamais E32*/E33*.
2. **La commande enregistrée de 2_305 n'est plus rejouable.** `queue/46_sub_2_305.done` = `--exp E34 --membres E32 E07 E08` : relancée
   aujourd'hui, elle prendrait le réseau (ordre identique car pris dans E34_test, mais probas différentes, moyenne 0,0371).
   Correction : régénérer `E32A_oof/test.csv` (09_verif_graines.py avec `--exp E32A --runs E30b E30bs7 E30bs2026`, ou mon recalcul)
   et noter la commande corrigée dans le rapport 11.
3. **`comparaisons.csv` garde deux lignes « E32 » de sens différent** (E32 vs E29m = piste A ; E32 vs E09 = réseau) et E36 est
   comparé « vs E32 » (piste A). `03_train.comparer` lit `<ref>_folds_detail.csv` par identifiant : toute future comparaison
   « vs E32 » utiliserait le réseau. Correction : renommer en E32A/E33A les lignes de la piste A dans `comparaisons.csv` et
   imposer des préfixes d'identifiants par piste.

**MINEUR**
4. Ex-aequo dans les mélanges de rangs test (17 956 lignes pour E34, 10 447 pour E33f) : `07_rangs_vers_proba` les départage par ordre
   d'id (tri stable). Effet sur le Gini négligeable ; c'est la seule source des écarts ≤ 1e-3 entre valeurs recalculées et fichiers.
5. Réseau sur GPU avec TF32 : graines fixées mais résultats non bit à bit reproductibles (cuDNN non déterministe). Acceptable.
6. Commentaire obsolète dans `12_dae.py` (« 1,49 M x F », « batch 1024 » alors que 595 k lignes et batch 512 utilisé).
7. La variance d'entraînement (autre découpage de folds) n'a pas été vérifiée pour le réseau (non vérifié par moi, voir §4).

## Verdict

**VALIDÉ AVEC RÉSERVES** (réserves = reproductibilité, aucune n'affecte les fichiers).
- `2_305.csv` : **DÉPOSABLE**.
- `2_306.csv` : **DÉPOSABLE, à privilégier** (+0,00105 vs E34, 5/5, IC90 bootstrap [+0,0004 ; +0,0016]).
- Chiffre : Gini CV 0,285 ± 0,007, attendu ≈ 0,284. Mentionner au jury que le DAE a été appris sur train + test sans la cible.
