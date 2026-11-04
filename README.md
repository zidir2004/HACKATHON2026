# Hackathon Data Science M2 DS4SC — Prédiction de sinistres (4 novembre 2026)

Objectif : prédire la probabilité qu'un assuré automobile déclare un sinistre (3,6 % de positifs).
Métrique : coefficient de Gini = 2·AUC − 1. Jeu de données inspiré du challenge public Kaggle « Porto Seguro ».

## Organisation
- `usine/00_brief.md` — compréhension du sujet
- `usine/JOURNAL.md` — **journal de bord** : décisions, protocole, toutes les expériences (y compris les échecs), revues
- `usine/src/` — code numéroté : `01_audit.py`, `02_prepare.py`, `03_train.py` (entraînement paramétrable), `06_melange.py`, `07_rangs_vers_proba.py`, `utils.py`…
- `usine/experiences/registre.csv` — une ligne par expérience ; `comparaisons.csv` — gains appariés fold par fold ; `folds.csv` — découpage figé
- `usine/rapports/` — un rapport par mission (audit, préparation, itérations, revues, analyse)
- `usine/figures/` — figures d'analyse ; `usine/soutenance/` — plan et questions-réponses

## Soumission finale 2_306
Modèle **E33f** : moyenne des rangs à poids égaux de 4 modèles — LightGBM (E30, paramètres Jahrer + one-hot),
XGBoost GPU (E07), CatBoost GPU (E08) et un réseau sur les activations d'un autoencodeur débruiteur (E32c, 3 graines) —
puis remise à l'échelle des probabilités. **Gini CV 0,28531 ± 0,0074, Gini serveur 0,30079.**

Schéma de la chaîne, rôle de chaque fichier, **commandes exactes dans l'ordre** et fichiers à montrer :
voir [`usine/src/README_2_306.md`](usine/src/README_2_306.md).

## Protocole
Validation croisée stratifiée 5 folds (graine 42, folds figés), Gini par fold, comparaison appariée à la référence,
une seule modification par expérience, revue indépendante avant chaque soumission.

Les données (`train.csv`, `test.csv`) ne sont pas publiées dans ce dépôt.
