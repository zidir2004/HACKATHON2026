"""20_selection_segments.py — sélection de modèle par segment (E95–E99).

Idée : « dans le segment A, XGBoost est le meilleur -> on l'utilise ; dans le segment B, le réseau
de neurones est meilleur -> on l'utilise ». Aucun réentraînement : on réutilise les OOF / tests des
4 membres de 2_306 (E30 LightGBM, E07 XGBoost, E08 CatBoost, E32c réseau DAE) + leur mélange E33f.

Protocole (emboîté, honnête) :
  1. Échelle commune : chaque candidat -> rang percentile global (OOF d'un côté, test de l'autre).
  2. Segmentations fixées a priori (segments >= 15 % des lignes, fusion automatique sinon) :
       E95 nb de valeurs manquantes par ligne (0-1 / 2 / >=3)
       E96 tertiles de ps_car_13 (seuils appris sur le train)
       E97 ps_ind_05_cat (-1 / 0 / autre)
  3. Pour chaque fold k : sur les 4 autres folds, Gini de chaque candidat dans chaque segment ->
     choix du meilleur par segment ; ce choix est appliqué au fold k. Gini GLOBAL du fold k
     comparé à E33f (apparié ; garde si 5/5 folds gagnés ou Δ > 2·sd/√5).
  4. E98 : variante douce sur la meilleure segmentation dure = moyenne des rangs des 2 meilleurs.
  5. E99 : meilleure variante, choix appris sur tout l'OOF, appliqué au test -> 2_311.csv
     (remise à l'échelle proba : ordre de la sélection, valeurs triées de la moyenne des probas
     des 4 membres).
Usage : python -u usine/src/20_selection_segments.py
"""
import pathlib, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

CANDIDATS = ["E30", "E07", "E08", "E32c", "E33f"]
NOMS = {"E30": "LightGBM", "E07": "XGBoost", "E08": "CatBoost", "E32c": "Réseau DAE", "E33f": "Mélange"}
MEMBRES_PROBA = ["E30", "E07", "E08", "E32b"]  # E32c_test = rangs moyens (pas des probas) ; même choix que 2_306
SORTIE = U.USINE / "submissions" / "2_311.csv"
FIG = U.USINE / "figures" / "12_selection_segments.png"
MIN_PART = 0.15


def gini_rapide(y, s):
    """Gini = 2*AUC-1 via les rangs (formule de Mann-Whitney), identique à U.gini."""
    r = pd.Series(s).rank(method="average").to_numpy()
    y = np.asarray(y)
    n1 = y.sum(); n0 = len(y) - n1
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return 2 * auc - 1


def pct(x):
    """Rang percentile global (échelle commune entre modèles)."""
    return pd.Series(x).rank(pct=True, method="average").to_numpy()


# ---------------------------------------------------------------- chargement
def charger():
    oof, tst = {}, {}
    for c in CANDIDATS:
        o = pd.read_csv(U.OOF_DIR / f"{c}_oof.csv").sort_values(U.ID, ignore_index=True)
        t = pd.read_csv(U.OOF_DIR / f"{c}_test.csv").sort_values(U.ID, ignore_index=True)
        oof[c], tst[c] = o, t
    base = oof["E30"]
    for c in CANDIDATS:
        assert (oof[c][U.ID].to_numpy() == base[U.ID].to_numpy()).all()
        assert (tst[c][U.ID].to_numpy() == tst["E30"][U.ID].to_numpy()).all()
    train, test = U.charger_donnees("v1")
    assert (train[U.ID].to_numpy() == base[U.ID].to_numpy()).all()
    fold = U.charger_folds(train)
    assert (fold == base["fold"].to_numpy()).all(), "folds différents de folds.csv"
    y = train[U.TARGET].to_numpy()
    R = {c: pct(oof[c]["oof"].to_numpy()) for c in CANDIDATS}           # rangs OOF
    T = {c: pct(tst[c]["prediction"].to_numpy()) for c in CANDIDATS}    # rangs test
    probas_test = np.mean([pd.read_csv(U.OOF_DIR / f"{c}_test.csv").sort_values(U.ID, ignore_index=True)
                           ["prediction"].to_numpy() for c in MEMBRES_PROBA], axis=0)
    return train, test, y, fold, R, T, probas_test, tst["E30"][U.ID].to_numpy()


# ---------------------------------------------------------------- segmentations (a priori)
def fusionner_petits(codes, n_seg):
    """Fusionne un segment < 15 % avec son voisin le plus petit (segments ordonnés)."""
    codes = codes.copy()
    while True:
        parts = np.array([(codes == s).mean() for s in range(n_seg)])
        if n_seg <= 2 or parts.min() >= MIN_PART:
            return codes, n_seg
        i = int(parts.argmin())
        j = i + 1 if i == 0 else (i - 1 if i == n_seg - 1 else (i - 1 if parts[i - 1] < parts[i + 1] else i + 1))
        lo, hi = min(i, j), max(i, j)
        codes[codes == hi] = lo
        codes[codes > hi] -= 1
        n_seg -= 1


def segmenter(train, test, feats):
    """Renvoie {exp: (codes_train, codes_test, noms_segments)}."""
    out = {}
    both = pd.concat([train[feats], test[feats]], ignore_index=True)
    nmiss = (both.isna() | (both == -1)).sum(axis=1).to_numpy()
    c = np.clip(nmiss - 1, 0, 2)                         # 0-1 -> 0 ; 2 -> 1 ; >=3 -> 2
    ntr = len(train)
    ctr, k = fusionner_petits(c[:ntr], 3)
    noms = {3: ["manq 0-1", "manq 2", "manq >=3"]}.get(k, [f"seg{i}" for i in range(k)])
    # si fusion, on reconstruit les noms à partir des bornes réelles
    if k != 3:
        noms = [f"manq {int(nmiss[:ntr][ctr == s].min())}-{int(nmiss[:ntr][ctr == s].max())}" for s in range(k)]
    cte = np.array([ctr[np.where(c[:ntr] == v)[0][0]] if (c[:ntr] == v).any() else 0 for v in c[ntr:]]) \
        if k != 3 else c[ntr:]
    out["E95"] = (ctr, cte, noms)

    v = train["ps_car_13"].to_numpy(dtype=float)
    q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
    f = lambda x: np.where(x <= q1, 0, np.where(x <= q2, 1, 2))
    out["E96"] = (f(v), f(test["ps_car_13"].to_numpy(dtype=float)),
                  [f"car_13 <= {q1:.3f}", f"car_13 {q1:.3f}-{q2:.3f}", f"car_13 > {q2:.3f}"])

    g = lambda s: np.where(s.isna() | (s == -1), 0, np.where(s == 0, 1, 2))
    a, b = g(train["ps_ind_05_cat"]), g(test["ps_ind_05_cat"])
    a2, k = fusionner_petits(a, 3)
    if k == 3:
        out["E97"] = (a, b, ["ind_05 = -1", "ind_05 = 0", "ind_05 autre"])
    else:  # -1 trop rare -> fusion attendue avec 0 : {-1,0} vs autre
        m = {0: a2[a == 0][0] if (a == 0).any() else 0, 1: a2[a == 1][0], 2: a2[a == 2][0]}
        nm = ["ind_05 -1/0", "ind_05 autre"] if k == 2 and m[0] == m[1] else [f"seg{i}" for i in range(k)]
        out["E97"] = (a2, np.vectorize(m.get)(b), nm)
    return out


# ---------------------------------------------------------------- évaluation emboîtée
def choisir(y, R, seg, masque, n_seg, top=1, cands=CANDIDATS):
    """Pour chaque segment : candidats classés par Gini sur les lignes `masque`."""
    choix, ginis = [], []
    for s in range(n_seg):
        m = masque & (seg == s)
        gs = {c: gini_rapide(y[m], R[c][m]) for c in cands}
        ordre = sorted(gs, key=gs.get, reverse=True)
        choix.append(ordre[:top]); ginis.append(gs)
    return choix, ginis


def composer(R, seg, choix):
    score = np.zeros(len(seg))
    for s, cs in enumerate(choix):
        m = seg == s
        score[m] = np.mean([R[c][m] for c in cs], axis=0)
    return score


def evaluer(y, fold, R, seg, n_seg, top, cands=CANDIDATS):
    oof = np.zeros(len(y)); gk, choix_k = [], []
    for k in range(U.N_FOLDS):
        choix, _ = choisir(y, R, seg, fold != k, n_seg, top, cands)
        te = fold == k
        sc = composer(R, seg, choix)
        oof[te] = sc[te]
        gk.append(gini_rapide(y[te], sc[te])); choix_k.append(choix)
    return oof, np.array(gk), choix_k


def verdict(d):
    if np.allclose(d, 0):
        return "neutre", 0          # sélection identique à la référence
    sd = d.std(ddof=1)
    gagnes = int((d > 0).sum())
    if gagnes == 5 or d.mean() > 2 * sd / np.sqrt(5):
        return "garde", gagnes
    if gagnes == 0 or d.mean() < -2 * sd / np.sqrt(5):
        return "rejete", gagnes
    return "neutre", gagnes


def main():
    train, test, y, fold, R, T, probas_test, ids_test = charger()
    feats = U.liste_features(train)
    segs = segmenter(train, test, feats)
    ref = np.array([gini_rapide(y[fold == k], R["E33f"][fold == k]) for k in range(U.N_FOLDS)])
    print(f"E33f (réf) Gini par fold {np.round(ref, 5)} moy {ref.mean():.5f} ± {ref.std(ddof=1):.5f}")
    res = {}
    for exp, (ctr, cte, noms) in segs.items():
        n_seg = len(noms)
        print(f"\n=== {exp} segments {noms} parts train {[round((ctr == s).mean(), 3) for s in range(n_seg)]}"
              f" test {[round((cte == s).mean(), 3) for s in range(n_seg)]}")
        oof, gk, ck = evaluer(y, fold, R, ctr, n_seg, 1)
        d = gk - ref
        v, ga = verdict(d)
        stab = [[ck[k][s][0] for k in range(5)] for s in range(n_seg)]
        print(f"Gini/fold {np.round(gk, 5)} moy {gk.mean():.5f} ± {gk.std(ddof=1):.5f} | Δ {d.mean():+.5f} "
              f"(sd {d.std(ddof=1):.5f}) folds gagnés {ga}/5 -> {v}")
        for s in range(n_seg):
            print(f"   {noms[s]:<26} choix par fold : {stab[s]}")
        # diagnostic (sans ID) : sélection parmi les 4 modèles seuls, sans le mélange
        _, g4, c4 = evaluer(y, fold, R, ctr, n_seg, 1, ["E30", "E07", "E08", "E32c"])
        print(f"   [diag sans mélange] Gini moy {g4.mean():.5f} | Δ vs E33f {(g4 - ref).mean():+.5f} | "
              f"choix {[[c4[k][s][0] for k in range(5)] for s in range(n_seg)]}")
        res[exp] = dict(oof=oof, gk=gk, d=d, v=v, ga=ga, stab=stab, noms=noms, ctr=ctr, cte=cte, top=1)

    meilleur_dur = max(segs, key=lambda e: res[e]["gk"].mean())
    ctr, cte, noms = segs[meilleur_dur]
    oof, gk, ck = evaluer(y, fold, R, ctr, len(noms), 2)
    d = gk - ref; v, ga = verdict(d)
    stab = [["+".join(ck[k][s]) for k in range(5)] for s in range(len(noms))]
    print(f"\n=== E98 (doux, top-2, segmentation de {meilleur_dur}) Gini/fold {np.round(gk, 5)} moy {gk.mean():.5f}"
          f" | Δ {d.mean():+.5f} (sd {d.std(ddof=1):.5f}) folds gagnés {ga}/5 -> {v}")
    for s in range(len(noms)):
        print(f"   {noms[s]:<26} choix par fold : {stab[s]}")
    res["E98"] = dict(oof=oof, gk=gk, d=d, v=v, ga=ga, stab=stab, noms=noms, ctr=ctr, cte=cte, top=2)

    # OOF + registre
    for e, r in res.items():
        pd.DataFrame({U.ID: train[U.ID], U.TARGET: y, "oof": r["oof"], "fold": fold}).to_csv(
            U.OOF_DIR / f"{e}_oof.csv", index=False)
        lib = {"E95": "nb manquants/ligne", "E96": "tertiles ps_car_13", "E97": "ps_ind_05_cat",
               "E98": f"doux top-2, segmentation {meilleur_dur}"}[e]
        U.ajouter_registre(dict(
            id=e, donnees_version="v1", modele="selection_segment",
            changement=f"meilleur candidat par segment ({lib}) parmi E30/E07/E08/E32c/E33f, choix emboîté",
            score_cv_moyen=r["gk"].mean(), score_cv_std=r["gk"].std(ddof=1), delta_vs_ref=r["d"].mean(),
            verdict=r["v"], commentaire=f"réf E33f apparié; folds gagnés {r['ga']}/5; Δ/fold "
            f"{[round(x, 5) for x in r['d']]}; segments {r['noms']}; choix/fold {r['stab']}"))

    # E99 : meilleure variante, choix appris sur tout l'OOF -> test
    best = max(res, key=lambda e: round(res[e]["gk"].mean(), 6))  # égalité -> première (E95)
    r = res[best]
    choix, ginis = choisir(y, R, r["ctr"], np.ones(len(y), bool), len(r["noms"]), r["top"])
    print(f"\n=== E99 : variante retenue {best} ; choix sur tout l'OOF : "
          f"{dict(zip(r['noms'], ['+'.join(c) for c in choix]))}")
    score_test = composer(T, r["cte"], choix)
    ordre = np.argsort(score_test, kind="mergesort")
    pred = np.empty(len(score_test)); pred[ordre] = np.sort(probas_test)
    U.ecrire_soumission(ids_test, pred, SORTIE)
    U.verifier_soumission(SORTIE, y.mean())
    raw = open(SORTIE, "rb").read()
    assert b"\r" not in raw and raw.count(b"\n") == len(test), "fins de ligne / nb lignes"
    s306 = pd.read_csv(U.USINE / "submissions" / "2_306.csv", header=None, names=[U.ID, "p"])
    assert (s306[U.ID].to_numpy() == ids_test).all()
    rho = pd.Series(pred).rank().corr(pd.Series(s306["p"].to_numpy()).rank())
    rho33 = pd.Series(pred).rank().corr(pd.Series(T["E33f"]).rank())
    print(f"Spearman 2_311 vs 2_306 = {rho:.5f} ; vs E33f test = {rho33:.5f} ; moyenne {pred.mean():.5f}")
    pd.DataFrame({U.ID: ids_test, "prediction": score_test}).to_csv(U.OOF_DIR / "E99_test.csv", index=False)
    U.ajouter_registre(dict(
        id="E99", donnees_version="v1", modele="selection_segment", score_cv_moyen=r["gk"].mean(),
        score_cv_std=r["gk"].std(ddof=1), delta_vs_ref=r["d"].mean(), verdict="info",
        changement=f"soumission 2_311 : variante {best} apprise sur tout l'OOF, appliquée au test",
        commentaire=f"choix {dict(zip(r['noms'], ['+'.join(c) for c in choix]))}; verdict CV de {best} = {r['v']} "
        f"({r['ga']}/5); Spearman vs 2_306 {rho:.5f}; format OK 148803 lignes LF sans en-tête"))

    # figure : Gini de chaque candidat par segment (tout l'OOF), gagnant mis en évidence
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    couleurs = {"E30": "#4C78A8", "E07": "#F58518", "E08": "#54A24B", "E32c": "#B279A2", "E33f": "#555555"}
    for ax, e in zip(axes, ["E95", "E96", "E97"]):
        ctr_e, _, noms_e = segs[e]
        ch, gi = choisir(y, R, ctr_e, np.ones(len(y), bool), len(noms_e), 1)
        w = 0.16
        for j, c in enumerate(CANDIDATS):
            xs = np.arange(len(noms_e)) + (j - 2) * w
            vals = [gi[s][c] for s in range(len(noms_e))]
            bars = ax.bar(xs, vals, w, color=couleurs[c], label=NOMS[c] if e == "E95" else None)
            for s, b in enumerate(bars):
                if ch[s][0] == c:
                    b.set_edgecolor("black"); b.set_linewidth(2)
                    ax.text(b.get_x() + w / 2, b.get_height() + 0.003, "★", ha="center", fontsize=11)
        ax.set_xticks(range(len(noms_e)))
        ax.set_xticklabels([f"{n}\n({(ctr_e == s).mean():.0%})" for s, n in enumerate(noms_e)], fontsize=8)
        tag = f"{e} : Δ CV {res[e]['d'].mean():+.4f} ({res[e]['ga']}/5)"
        ax.set_title(tag, fontsize=10); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("Gini dans le segment (OOF)")
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Sélection du meilleur modèle par segment (★ = choisi)", y=1.06)
    fig.tight_layout(); fig.savefig(FIG, dpi=130, bbox_inches="tight")
    print("figure :", FIG)


if __name__ == "__main__":
    main()
