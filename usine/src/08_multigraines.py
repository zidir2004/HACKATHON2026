"""08_multigraines.py — moyenne multi-graines par membre (E17a/b/c) + diagnostics du mélange E17.

Usage :
  python -u usine/src/08_multigraines.py membres          # E17a/E17b/E17c à partir des graines 42, 7, 2026
  python -u usine/src/08_multigraines.py diagnostic --exp E17 --membres E17a E17b E17c --ref2 E09

membres : pour chaque membre de base (E13 LightGBM, E07 XGBoost, E08 CatBoost), la graine 42 est
  l'expérience d'origine (même code, mêmes paramètres) et les graines 7 et 2026 viennent des jobs
  <EXP>s7 / <EXP>s2026 (03_train.py --graines). Les PROBABILITÉS des 3 graines sont moyennées
  (même modèle, même échelle), OOF et test. Comparaison appariée vs le membre d'origine, ligne au registre.
diagnostic : comparaison appariée supplémentaire du mélange vs une 2e référence (ex. E09), et test
  des poids non égaux SANS biais : poids choisis sur 4 folds (grille pas 0,05), évalués sur le 5e.
"""
import argparse, importlib, itertools, pathlib, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
T = importlib.import_module("03_train")

BASES = {"E17a": ("E13", "lightgbm"), "E17b": ("E07", "xgboost"), "E17c": ("E08", "catboost")}
GRAINES = (7, 2026)


def lire(exp, quoi):
    return pd.read_csv(U.OOF_DIR / f"{exp}_{quoi}.csv").sort_values(U.ID, ignore_index=True)


def membres(quels=None):
    for new, (base, modele) in BASES.items():
        if quels and new not in quels:
            continue
        runs = [base] + [f"{base}s{g}" for g in GRAINES]
        oofs, tests = [lire(r, "oof") for r in runs], [lire(r, "test") for r in runs]
        y, fold = oofs[0][U.TARGET].to_numpy(), oofs[0]["fold"].to_numpy()
        assert all((o["fold"].to_numpy() == fold).all() for o in oofs), "folds différents !"
        P = np.column_stack([o["oof"] for o in oofs])
        p, pt = P.mean(1), np.mean([t["prediction"] for t in tests], axis=0)
        gs = [U.gini(y[fold == k], p[fold == k]) for k in range(U.N_FOLDS)]
        par_graine = [np.mean([U.gini(y[fold == k], P[fold == k, j]) for k in range(5)]) for j in range(3)]
        corr = np.corrcoef(P.T)[np.triu_indices(3, 1)]
        print(f"\n== {new} = moyenne 3 graines de {base} ({modele})\n  gini CV par graine {np.round(par_graine, 5)} "
              f"(écart max {max(par_graine) - min(par_graine):.5f}) ; corr. OOF entre graines {np.round(corr, 4)}")
        print(f"  => gini CV {np.mean(gs):.5f} ± {np.std(gs):.5f} | OOF {U.gini(y, p):.5f}")
        pd.DataFrame({U.ID: oofs[0][U.ID], U.TARGET: y, "oof": p, "fold": fold}).to_csv(U.OOF_DIR / f"{new}_oof.csv", index=False)
        pd.DataFrame({U.ID: tests[0][U.ID], "prediction": pt}).to_csv(U.OOF_DIR / f"{new}_test.csv", index=False)
        det = [pd.read_csv(U.OOF_DIR / f"{r}_folds_detail.csv").sort_values("fold", ignore_index=True) for r in runs]
        duree = sum(d["duree_s"].sum() for d in det)
        pd.DataFrame({"n_arbres": [str([int(d.loc[k, "n_arbres"]) for d in det]) for k in range(5)], "fold": range(5),
                      "gini": gs, "duree_s": [sum(d.loc[k, "duree_s"] for d in det) for k in range(5)]}).to_csv(
            U.OOF_DIR / f"{new}_folds_detail.csv", index=False)
        c = T.comparer(new, base, gs)
        print(f"  apparié vs {base} : Δmoy={c['d_moy']:+.5f} folds gagnés {c['folds_gagnes']} "
              f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} seuil 2se {c['seuil_2se']} -> {c['verdict']}")
        U.ajouter_registre(dict(
            id=new, donnees_version="v1", modele=modele, verdict=c["verdict"], delta_vs_ref=c["d_moy"],
            changement=f"{base} moyenné sur 3 graines de modèle (42, 7, 2026), mêmes folds",
            score_cv_moyen=float(np.mean(gs)), score_cv_std=float(np.std(gs)),
            commentaire=(f"gini OOF {U.gini(y, p):.5f}; gini CV par graine {np.round(par_graine, 5).tolist()}; "
                         f"{duree:.0f}s cumulés (3 graines); apparié vs {base}: Δmoy {c['d_moy']:+.5f} (std {c['d_std']:.5f}), "
                         f"folds gagnés {c['folds_gagnes']}, Δ/fold {[c[f'd_fold{k}'] for k in range(5)]}")))


def diagnostic(exp, mbs, ref2):
    if ref2:
        g_exp = pd.read_csv(U.OOF_DIR / f"{exp}_folds_detail.csv").sort_values("fold")["gini"].to_numpy()
        c = T.comparer(exp, ref2, g_exp)
        print(f"\n== {exp} apparié vs {ref2} : Δmoy={c['d_moy']:+.5f} (std {c['d_std']:.5f}) folds gagnés {c['folds_gagnes']} "
              f"Δ/fold {[c[f'd_fold{k}'] for k in range(5)]} seuil 2se {c['seuil_2se']} -> {c['verdict']}")
    oofs = [lire(m, "oof") for m in mbs]
    y, fold = oofs[0][U.TARGET].to_numpy(), oofs[0]["fold"].to_numpy()
    R = np.column_stack([rankdata(o["oof"]) / len(o) for o in oofs])
    grille = [np.array(w) for w in itertools.product(np.arange(0, 1.001, 0.05), repeat=len(mbs)) if abs(sum(w) - 1) < 1e-9]
    eg = np.ones(len(mbs)) / len(mbs)
    print(f"  poids non égaux, choisis sur 4 folds / évalués sur le 5e (estimation sans biais) :")
    d = []
    for k in range(5):
        tr, va = fold != k, fold == k
        # Gini moyen par fold (pas Gini OOF global) pour choisir les poids sur les 4 folds
        sc = [np.mean([U.gini(y[fold == j], R[fold == j] @ w) for j in range(5) if j != k]) for w in grille]
        w = grille[int(np.argmax(sc))]
        g_w, g_eg = U.gini(y[va], R[va] @ w), U.gini(y[va], R[va] @ eg)
        d.append(g_w - g_eg)
        print(f"    fold {k} : poids {np.round(w, 2).tolist()} -> gini {g_w:.5f} vs égaux {g_eg:.5f} (Δ {g_w - g_eg:+.5f})")
    d = np.array(d)
    print(f"  => Δ poids appris - poids égaux : {d.mean():+.5f} (sd {d.std(ddof=1):.5f}, {int((d > 0).sum())}/5 folds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["membres", "diagnostic"])
    ap.add_argument("--exp"); ap.add_argument("--membres", nargs="+"); ap.add_argument("--ref2"); ap.add_argument("--quels", nargs="*")
    a = ap.parse_args()
    membres(a.quels) if a.mode == "membres" else diagnostic(a.exp, a.membres, a.ref2)
