"""09_verif_graines.py — vérification de robustesse d'un changement de features à la graine du modèle.

Usage : python -u usine/src/09_verif_graines.py --exp E28 --runs E27 E27s7 E27s2026 \
            --ref E02m --ref-runs E02 E02s7 E02s2026

Pourquoi : un gain apparié sur 5 folds avec UNE graine de modèle mélange l'effet du changement et le
bruit de graine (sous-échantillonnage de lignes/colonnes LightGBM). On moyenne les PROBABILITÉS des
3 graines de chaque côté (mêmes folds, mêmes paramètres) puis on compare apparié par fold.
La référence moyennée (<ref>) est sauvegardée comme ingrédient (hors registre) ; <exp> va au registre.
"""
import argparse, importlib, pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
T = importlib.import_module("03_train")

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True); ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--ref", required=True); ap.add_argument("--ref-runs", nargs="+", required=True)
ap.add_argument("--changement", default="")
ap.add_argument("--version", default="v4")
a = ap.parse_args()


def moyenne(nom, runs):
    oofs = [pd.read_csv(U.OOF_DIR / f"{r}_oof.csv").sort_values(U.ID, ignore_index=True) for r in runs]
    tests = [pd.read_csv(U.OOF_DIR / f"{r}_test.csv").sort_values(U.ID, ignore_index=True) for r in runs]
    y, fold = oofs[0][U.TARGET].to_numpy(), oofs[0]["fold"].to_numpy()
    assert all((o["fold"].to_numpy() == fold).all() for o in oofs), "folds différents !"
    p = np.mean([o["oof"] for o in oofs], axis=0)
    gs = [U.gini(y[fold == k], p[fold == k]) for k in range(U.N_FOLDS)]
    pd.DataFrame({U.ID: oofs[0][U.ID], U.TARGET: y, "oof": p, "fold": fold}).to_csv(U.OOF_DIR / f"{nom}_oof.csv", index=False)
    pd.DataFrame({U.ID: tests[0][U.ID], "prediction": np.mean([t["prediction"] for t in tests], axis=0)}).to_csv(
        U.OOF_DIR / f"{nom}_test.csv", index=False)
    pd.DataFrame({"n_arbres": "n/a", "fold": range(5), "gini": gs, "duree_s": 0}).to_csv(
        U.OOF_DIR / f"{nom}_folds_detail.csv", index=False)
    print(f"  {nom} = moyenne de {runs} : gini CV {np.mean(gs):.5f} ± {np.std(gs):.5f} | OOF {U.gini(y, p):.5f}")
    return gs, U.gini(y, p)


moyenne(a.ref, a.ref_runs)
gs, g_oof = moyenne(a.exp, a.runs)
c = T.comparer(a.exp, a.ref, gs)
print(f"  apparié vs {a.ref} : Δmoy={c['d_moy']:+.5f} folds gagnés {c['folds_gagnes']} "
      f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} seuil 2se {c['seuil_2se']} -> {c['verdict']}")
U.ajouter_registre(dict(
    id=a.exp, donnees_version=a.version, modele="lightgbm", verdict=c["verdict"], delta_vs_ref=c["d_moy"],
    changement=a.changement or f"{'+'.join(a.runs)} (moyenne 3 graines) vs {a.ref}",
    score_cv_moyen=float(np.mean(gs)), score_cv_std=float(np.std(gs)),
    commentaire=(f"gini OOF {g_oof:.5f}; apparié vs {a.ref} (= moyenne {'+'.join(a.ref_runs)}): Δmoy {c['d_moy']:+.5f}, "
                 f"folds gagnés {c['folds_gagnes']}, seuil 2se {c['seuil_2se']}, Δ/fold {[c[f'd_fold{k}'] for k in range(5)]}")))
