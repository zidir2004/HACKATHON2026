"""13_nn_dae.py — Piste B, étape 2 : réseau supervisé sur les activations du DAE (expérience E32).

Source : solution gagnante Porto Seguro (Michael Jahrer, 1re place, Kaggle 2017).
Entrée du réseau = concaténation des 3 couches cachées du DAE gelé (3 x 1500 = 4500 valeurs),
calculées À LA VOLÉE par lots sur le GPU (595k x 4500 en float32 = 10,7 Go : impossible à stocker).
Réseau : 4500-1000-1000-1, ReLU, dropout d'entrée 0,1, dropout 0,5, sortie sigmoïde, log-loss.

Protocole : folds figés usine/experiences/folds.csv (obligatoire). Un réseau par fold, early stopping
sur le Gini du fold de validation -> léger biais optimiste (l'époque est choisie sur le fold évalué ;
mesuré dans le rapport via le Gini à époque commune). Prédiction test = moyenne des 5 réseaux.

Rôle dans la chaîne 2_306 : produit le 4e membre du mélange (E32c = moyenne de 3 graines).
  Version retenue = --moyenne-epoques 6 (E32b) : 6 époques FIXES et moyenne des prédictions des
  6 époques, SANS early stopping -> évite de choisir l'arrêt sur le fold évalué (score CV honnête).
  Commandes exactes (GPU, ~2-3 min chacune, log usine/logs/39_nn_E32b.log, 42_*, 43_*) :
    python -u usine/src/13_nn_dae.py --exp E32b    --moyenne-epoques 6
    python -u usine/src/13_nn_dae.py --exp E32b_g1 --moyenne-epoques 6 --graine 1
    python -u usine/src/13_nn_dae.py --exp E32b_g2 --moyenne-epoques 6 --graine 2
  Entrées : usine/data/dae/X_all.npy, meta.json, dae.pt (12_dae.py), folds.csv, données v1 (cible),
            usine/experiences/oof/E09_oof.csv (référence de comparaison seulement)
  Sorties : usine/experiences/oof/<exp>_oof.csv, _test.csv, _folds_detail.csv + ligne au registre.

Usage : python -u usine/src/13_nn_dae.py --exp E32 [--folds 0 1 2 3 4] [--sans-dae]
  Chaque fold est sauvé dans usine/data/dae/<exp>_fold<k>.npz ; quand les 5 existent, le script
  assemble OOF/test (oof/<exp>_oof.csv, _test.csv, _folds_detail.csv), compare à E09 et écrit le registre.
  --sans-dae : solution de repli = même réseau directement sur les entrées RankGauss/one-hot.
"""
import argparse
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
DAE_MOD = importlib.import_module("12_dae")
DAE_DIR = DAE_MOD.DAE_DIR
torch.set_num_threads(12)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"


# ---------------------------------------------------------------------------
# 1. Chargement : entrées préparées par 12_dae.py, cible, folds figés, DAE gelé
# ---------------------------------------------------------------------------
def charger(sans_dae: bool, fichier_dae: str = "dae.pt"):
    """Charge X (train+test préparés par 12_dae.py) sur le GPU, la cible, les folds figés et le DAE gelé.

    Le DAE est GELÉ (requires_grad False) : il sert seulement d'extracteur de représentation ;
    seul le petit réseau supervisé apprend avec la cible, dans les folds d'entraînement."""
    meta = json.loads((DAE_DIR / "meta.json").read_text(encoding="utf-8"))
    X = torch.from_numpy(np.load(DAE_DIR / "X_all.npy")).to(DEV)
    train, test = U.charger_donnees("v1")                         # même tri par id que 12_dae.py
    # Garde-fou : les lignes de X_all.npy doivent être dans le même ordre que la cible (tri par id)
    assert len(train) == meta["n_train"] and int(train[U.ID].iloc[0]) == meta["ids_train_premier"]
    y = train[U.TARGET].to_numpy(dtype=np.float32)
    fold = U.charger_folds(train)
    dae = None
    if not sans_dae:
        ck = torch.load(DAE_DIR / fichier_dae, map_location=DEV)
        dae = DAE_MOD.DAE(ck["n_in"], ck.get("n_hid", 1500)).to(DEV); dae.load_state_dict(ck["etat"]); dae.eval()
        for p in dae.parameters():
            p.requires_grad_(False)
        print(f"[1] DAE chargé ({ck['epoques']} époques)", flush=True)
    return X, y, fold, train[U.ID].to_numpy(), test[U.ID].to_numpy(), meta["n_train"], dae


def representation(dae, xb):
    """Entrée du réseau : activations concaténées des 3 couches cachées (ou x brut en repli)."""
    if dae is None:
        return xb
    with torch.no_grad():
        return torch.cat(dae.activations(xb), dim=1)


# ---------------------------------------------------------------------------
# 2. Réseau supervisé
# ---------------------------------------------------------------------------
class Reseau(nn.Module):
    """4500 -> 1000 -> 1000 -> 1. Dropout fort (0,5) : 4500 entrées pour seulement ~16 000 sinistres
    au total, le réseau sur-apprendrait vite sans régularisation."""
    def __init__(self, n_in, n_hid=1000, drop_in=0.1, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop_in), nn.Linear(n_in, n_hid), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(n_hid, n_hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(n_hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)            # logit ; la sigmoïde est dans la perte


def predire(model, dae, X, idx, batch=4096):
    """Probabilités (sigmoïde) pour les lignes idx, calculées par lots (représentation DAE à la volée)."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            out.append(torch.sigmoid(model(representation(dae, X[idx[i:i + batch]]))).float().cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# 3. Un fold : entraînement + early stopping sur le Gini du fold de validation
# ---------------------------------------------------------------------------
def un_fold(k, X, y, fold, n_train, dae, a):
    """Entraîne un réseau sur les 4 folds d'entraînement, prédit le fold k (OOF) et tout le test.
    Sauvegarde usine/data/dae/<exp>_fold<k>.npz (permet de relancer un fold seul)."""
    torch.manual_seed(U.SEED + k + 1000 * a.graine)   # graine 0 = E32/E32b
    # Indices : entraînement = autres folds ; validation = fold k ; test = lignes après n_train dans X
    i_tr = torch.from_numpy(np.where(fold != k)[0]).to(DEV)
    i_va = np.where(fold == k)[0]
    i_va_g, i_te = torch.from_numpy(i_va).to(DEV), torch.arange(n_train, X.shape[0], device=DEV)
    yg = torch.from_numpy(y).to(DEV)
    n_in = representation(dae, X[:2]).shape[1]
    model = Reseau(n_in).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    meilleur, etat, ep_best, courbe, t0 = -1.0, None, 0, [], time.time()
    # Variante sans biais (--moyenne-epoques N) : N époques FIXES, prédiction = moyenne des prédictions
    # de chaque époque (« snapshot ensemble ») -> aucune décision prise sur le fold de validation.
    somme_va, somme_te = 0.0, 0.0
    n_ep = a.moyenne_epoques or a.max_epochs
    for ep in range(1, n_ep + 1):
        te = time.time(); model.train()
        perm = i_tr[torch.randperm(len(i_tr), device=DEV)]
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            # Log-loss sur les logits (plus stable numériquement que sigmoïde + BCE séparées)
            perte = nn.functional.binary_cross_entropy_with_logits(model(representation(dae, X[b])), yg[b])
            opt.zero_grad(set_to_none=True); perte.backward(); opt.step()
        p_va = predire(model, dae, X, i_va_g)
        g = U.gini(y[i_va], p_va); courbe.append(round(g, 5))
        # Mode retenu pour 2_306 : on cumule les prédictions de chaque époque (val ET test)
        if a.moyenne_epoques:
            somme_va, somme_te = somme_va + p_va, somme_te + predire(model, dae, X, i_te)
            g_moy = U.gini(y[i_va], somme_va)
            print(f"  fold {k} ep {ep:2d} | {time.time()-te:5.1f}s | gini val {g:.5f} | moyenne ep 1..{ep} {g_moy:.5f}",
                  flush=True)
            continue
        # Mode early stopping (E32, non retenu) : on garde les poids de la meilleure époque
        if g > meilleur:
            meilleur, ep_best, etat = g, ep, {n: t.detach().clone() for n, t in model.state_dict().items()}
        print(f"  fold {k} ep {ep:2d} | {time.time()-te:5.1f}s | gini val {g:.5f} | meilleur {meilleur:.5f} (ep {ep_best})",
              flush=True)
        if ep - ep_best >= a.patience or time.time() - t0 > a.budget_fold_min * 60:
            break
    if a.moyenne_epoques:
        p_va, p_te, ep_best = somme_va / n_ep, somme_te / n_ep, n_ep
        meilleur = U.gini(y[i_va], p_va)
    else:
        model.load_state_dict(etat)
        p_va, p_te = predire(model, dae, X, i_va_g), predire(model, dae, X, i_te)
    np.savez(DAE_DIR / f"{a.exp}_fold{k}.npz", i_va=i_va, p_va=p_va, p_te=p_te, courbe=np.array(courbe),
             ep_best=ep_best, duree=time.time() - t0)
    print(f"[3] fold {k} : gini {meilleur:.5f} à l'époque {ep_best} | {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# 4. Assemblage des 5 folds : OOF, test, comparaison appariée vs E09, registre
# ---------------------------------------------------------------------------
def assembler(y, fold, ids, ids_test, a):
    """Réunit les 5 folds : OOF complet, prédiction test = moyenne des 5 réseaux, Gini par fold,
    comparaison appariée vs E09, ligne au registre."""
    T = importlib.import_module("03_train")
    oof, p_te, gs, details, courbes = np.full(len(y), np.nan), [], [], [], []
    for k in range(U.N_FOLDS):
        d = np.load(DAE_DIR / f"{a.exp}_fold{k}.npz")
        oof[d["i_va"]] = d["p_va"]; p_te.append(d["p_te"]); courbes.append(d["courbe"])
        gs.append(U.gini(y[d["i_va"]], d["p_va"]))
        details.append(dict(n_arbres=int(d["ep_best"]), fold=k, gini=gs[-1], duree_s=round(float(d["duree"]), 1)))
    assert not np.isnan(oof).any()
    # Biais de l'early stopping par fold : Gini de chaque fold à une époque COMMUNE (médiane des meilleures)
    ep_c = int(np.median([d["n_arbres"] for d in details]))
    g_c = [float(c[min(ep_c, len(c)) - 1]) for c in courbes]
    pd.DataFrame({U.ID: ids, U.TARGET: y.astype(int), "oof": oof, "fold": fold}).to_csv(U.OOF_DIR / f"{a.exp}_oof.csv", index=False)
    pd.DataFrame({U.ID: ids_test, "prediction": np.mean(p_te, axis=0)}).to_csv(U.OOF_DIR / f"{a.exp}_test.csv", index=False)
    pd.DataFrame(details).to_csv(U.OOF_DIR / f"{a.exp}_folds_detail.csv", index=False)
    e09 = pd.read_csv(U.OOF_DIR / "E09_oof.csv").sort_values(U.ID, ignore_index=True)
    assert (e09[U.ID].to_numpy() == ids).all()
    rho = spearmanr(oof, e09["oof"]).correlation
    c = T.comparer(a.exp, "E09", gs)
    print(f"[4] {a.exp} : gini CV {np.mean(gs):.5f} ± {np.std(gs):.5f} | OOF {U.gini(y, oof):.5f} | folds "
          f"{np.round(gs, 5).tolist()} | époques {[d['n_arbres'] for d in details]}\n"
          f"    à époque commune {ep_c} : {np.mean(g_c):.5f} ({'gain de la moyenne des époques vs dernière époque' if a.moyenne_epoques else 'biais early stopping'} ≈ {np.mean(gs)-np.mean(g_c):+.5f})\n"
          f"    Spearman OOF vs E09 = {rho:.4f} | apparié vs E09 Δ {c['d_moy']:+.5f} ({c['folds_gagnes']})", flush=True)
    U.ajouter_registre(dict(
        id=a.exp, donnees_version="v1 (one-hot + RankGauss)", modele="nn_dae" if not a.sans_dae else "mlp",
        changement=((f"réseau 4500-1000-1000 sur activations DAE swap-noise (Jahrer réduit, {a.dae})" + (
                    f", SANS early stopping : moyenne des prédictions des époques 1..{a.moyenne_epoques}"
                    if a.moyenne_epoques else ", early stopping par fold")) if not a.sans_dae
                    else "MLP 1000-1000 directement sur entrées RankGauss/one-hot (repli sans DAE)"),
        score_cv_moyen=float(np.mean(gs)), score_cv_std=float(np.std(gs)), delta_vs_ref=c["d_moy"],
        verdict="rejete" if c["verdict"] == "rejete" else "info",
        commentaire=(f"modèle seul, pour mélange (E33); gini OOF {U.gini(y, oof):.5f}; époques/fold "
                     f"{[d['n_arbres'] for d in details]}; Gini à époque commune {ep_c} : {np.mean(g_c):.5f}; "
                     f"Spearman OOF vs E09 {rho:.4f}; apparié vs E09 Δ {c['d_moy']:+.5f} ({c['folds_gagnes']})")))


# ---------------------------------------------------------------------------
# 5. Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E32")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--sans-dae", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5, help="L2 (weight_decay d'Adam)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--budget-fold-min", type=float, default=11)
    ap.add_argument("--dae", default="dae.pt", help="checkpoint du DAE dans usine/data/dae")
    ap.add_argument("--graine", type=int, default=0, help="décalage de graine (variantes multi-graines)")
    ap.add_argument("--moyenne-epoques", type=int, default=0,
                    help="N>0 : N époques fixes, moyenne des prédictions par époque (pas d'early stopping)")
    a = ap.parse_args()
    X, y, fold, ids, ids_test, n_train, dae = charger(a.sans_dae, a.dae)
    for k in a.folds:
        un_fold(k, X, y, fold, n_train, dae, a)
    # Assemblage seulement quand les 5 folds existent (les folds peuvent être lancés séparément)
    if all((DAE_DIR / f"{a.exp}_fold{k}.npz").exists() for k in range(U.N_FOLDS)):
        assembler(y, fold, ids, ids_test, a)
