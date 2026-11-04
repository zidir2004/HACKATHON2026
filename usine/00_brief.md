# Brief — Hackathon Data Science M2 DS4SC (4 novembre 2026)

## Sujet
- **Cible** : `target` = 1 si l'assuré déclare un sinistre (assurance auto).
- **Tâche** : classification binaire, **prédiction en probabilité** (pas de classe 0/1).
- **Métrique** : coefficient de **Gini = 2 × AUC − 1**. Seul l'ordre des prédictions compte : pas de seuil à optimiser, pas de calibration nécessaire.
- **Données** : train 446 409 lignes × 59 colonnes (id, target, 57 variables) ; test 148 803 lignes. Taux de sinistre : **3,64 %** (très déséquilibré).
- Identifiants train/test disjoints, test trié par id croissant, `submission_example.csv` aligné sur le test.

## Format de soumission (strict)
- CSV **sans en-tête**, 2 colonnes `id,prediction`, **148 803 lignes**, id croissants.
- Nom : `[idgroupe]_[idsoumission].csv` → groupe **2**, première soumission **2_302.csv** (puis numéros suivants, à confirmer).
- Dépôt : https://utbox.univ-tours.fr/s/mS8549YmdjmX4Tk (dépôt manuel par l'équipe).

## Barème (20 pts)
- Baseline /3 (dépasser la baseline du prof), compétition /4.
- **Oral /13** : méthodologie /4, analyse approfondie /4, clarté du code /2, travail en équipe /3.
→ La démarche, l'analyse des données et la lisibilité du code pèsent plus que les derniers millièmes de Gini.

## Horaires (heure de Paris)
- 11h et 14h : check-ups. **16h : fermeture des soumissions.** 16h30 : présentations.
- Gel des expériences risquées à 15h.

## Ce qu'on sait déjà
- Variables anonymisées par groupes : `ind` (individu), `reg` (région), `car` (véhicule), `calc` (variables calculées). Suffixes `_bin` (binaire), `_cat` (catégorielle), sinon continu/ordinal.
- **-1 = valeur manquante** (ex. `ps_car_03_cat` manquant à 69 %, `ps_car_05_cat` 45 %, `ps_reg_03` 18 %).
- Le jeu correspond au challenge public Kaggle « Porto Seguro Safe Driver Prediction » (2017). À dire ouvertement à l'oral. Repères connus : Gini ≈ 0,28–0,29 pour un bon modèle ; les `ps_calc_*` sont réputées n'apporter que du bruit — **on le vérifiera nous-mêmes** plutôt que de le supposer.

## Pièges pressentis
- Signal faible (Gini attendu < 0,30) → beaucoup de bruit entre folds : comparer les gains à l'écart-type.
- Manquants (-1) probablement informatifs : ne pas les imputer bêtement.
- Catégorielle à forte cardinalité `ps_car_11_cat` (104 modalités) : encodage cible uniquement dans les folds.
- Surapprentissage du leaderboard : on soumet seulement ce qui progresse en validation croisée.

## Environnement
- Calcul sur le **PC Windows de l'utilisateur** (OMEN), à sa demande. Le shell distant n'y a pas d'accès direct (VM limitée), donc : l'utilisateur lance une fois `python usine\runner.py` ; le lead-tech dépose des jobs dans `usine/queue/`, lit les logs dans `usine/logs/`.
- Dossier projet : `C:\Users\Pitchi\Desktop\Univ-M2\Hackathon\` (données d'origine à la racine, jamais modifiées).
