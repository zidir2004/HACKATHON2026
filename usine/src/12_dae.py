"""12_dae.py — Piste B, étape 1 : autoencodeur débruiteur (DAE) non supervisé.

Source : solution gagnante Kaggle « Porto Seguro's Safe Driver Prediction » (Michael Jahrer, 1re place,
post « 1st place with representation learning », forum Kaggle, 2017).
Idée : apprendre une représentation des assurés SANS la cible, en demandant à un réseau de reconstruire
une ligne dont 15 % des cases ont été « échangées » avec la même colonne d'une autre ligne (swap noise).
Les activations des couches cachées servent ensuite d'entrée à un réseau supervisé (13_nn_dae.py).

Version RÉDUITE (contrainte de temps du hackathon) :
  original : 221-1500-1500-1500-221, batch 128, lr 3e-3 décroissance 0,995/époque, 1000 époques (~5 h)
  ici      : même architecture, batch 1024, Adam, lr décroissant selon le TEMPS écoulé, budget ~35 min
             (le nombre d'époques réalisé est écrit dans le log et dans le checkpoint).

Le DAE est entraîné sur train + test (aucune cible utilisée : légitime, mais à signaler).

Usage : python -u usine/src/12_dae.py --budget-min 35 [--batch 1024] [--lr 1e-3]
Sorties : usine/data/dae/X_all.npy (entrées préparées, train puis test, triés par id),
          usine/data/dae/meta.json, usine/data/dae/dae.pt (poids), log par époque.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import erfinv
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

DAE_DIR = U.USINE / "data" / "dae"
DAE_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(12)                       # piste A garde ses 16 cœurs
torch.backends.cuda.matmul.allow_tf32 = True     # TF32 : ~2x plus rapide sur RTX 40xx, précision suffisante
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# 1. Préparation des entrées (recette Jahrer)
#    - ps_calc_* déjà retirées (données v1)
#    - variables _cat -> one-hot (le -1 « manquant » devient une modalité à part)
#    - binaires 0/1 inchangées
#    - numériques -> RankGauss : rang -> [-1, 1] -> erfinv  (distribution ~ gaussienne)
#      Le -1 manquant reste une valeur (le plus petit rang), comme dans l'original.
#    Calculé sur train + test (transformation non supervisée).
# ---------------------------------------------------------------------------
def rank_gauss(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average")                  # ex aequo -> même valeur
    u = (r - 0.5) / len(x) * 2.0 - 1.0                 # dans ]-1, 1[
    u = np.clip(u, -1 + 1e-6, 1 - 1e-6)
    z = erfinv(u)
    return (z - z.mean()).astype(np.float32)


def preparer_entrees():
    train, test = U.charger_donnees("v1")             # triés par id
    schema = json.loads((U.USINE / "data/processed/v1/schema.json").read_text(encoding="utf-8"))
    tout = pd.concat([train.drop(columns=[U.TARGET]), test], ignore_index=True)
    blocs, noms = [], []
    for c in schema["numeric_cols"]:
        blocs.append(rank_gauss(tout[c].to_numpy(dtype=float))[:, None]); noms.append(c)
    for c in schema["binary_cols"]:
        v = tout[c].to_numpy()
        if set(np.unique(v)) <= {0, 1}:
            blocs.append(v.astype(np.float32)[:, None]); noms.append(c)
        else:                                        # binaire avec -1 -> one-hot
            oh = pd.get_dummies(v, prefix=c).astype(np.float32)
            blocs.append(oh.to_numpy()); noms += list(oh.columns)
    for c in schema["cat_cols"]:
        oh = pd.get_dummies(tout[c], prefix=c).astype(np.float32)
        blocs.append(oh.to_numpy()); noms += list(oh.columns)
    X = np.hstack(blocs).astype(np.float32)
    meta = dict(n_train=len(train), n_test=len(test), n_features=X.shape[1], colonnes=noms,
                ids_train_premier=int(train[U.ID].iloc[0]), ids_test_premier=int(test[U.ID].iloc[0]))
    np.save(DAE_DIR / "X_all.npy", X)
    (DAE_DIR / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[1] entrées : {X.shape} (train {len(train)} + test {len(test)}), "
          f"{len(schema['numeric_cols'])} RankGauss, one-hot -> {X.shape[1]} colonnes", flush=True)
    return X


# ---------------------------------------------------------------------------
# 2. Modèle : « deep stack » F-1500-1500-1500-F, ReLU, sortie linéaire
# ---------------------------------------------------------------------------
class DAE(nn.Module):
    def __init__(self, n_in: int, n_hid: int = 1500):
        super().__init__()
        self.h1, self.h2, self.h3 = nn.Linear(n_in, n_hid), nn.Linear(n_hid, n_hid), nn.Linear(n_hid, n_hid)
        self.out = nn.Linear(n_hid, n_in)

    def activations(self, x):
        """Renvoie les 3 couches cachées (utilisées comme représentation par 13_nn_dae.py)."""
        a1 = torch.relu(self.h1(x)); a2 = torch.relu(self.h2(a1)); a3 = torch.relu(self.h3(a2))
        return a1, a2, a3

    def forward(self, x):
        return self.out(self.activations(x)[2])


# ---------------------------------------------------------------------------
# 3. Bruit par échange (swap noise) : chaque case est remplacée, avec proba p, par la valeur
#    de la MÊME colonne d'une ligne tirée au hasard dans tout le jeu (train + test).
#    Avantage vs bruit gaussien : la valeur bruitée reste une valeur plausible de la colonne.
# ---------------------------------------------------------------------------
def swap_noise(xb, X_all, p: float):
    B, F = xb.shape
    lignes = torch.randint(0, X_all.shape[0], (B, F), device=xb.device)
    cols = torch.arange(F, device=xb.device).expand(B, F)
    masque = torch.rand(B, F, device=xb.device) < p
    return torch.where(masque, X_all[lignes, cols], xb)


# ---------------------------------------------------------------------------
# 4. Entraînement borné dans le temps
#    lr(t) = lr0 * lr_fin^(t / budget) : décroissance exponentielle selon le temps écoulé,
#    de sorte que le programme se termine « proprement » quel que soit le nombre d'époques réalisé.
# ---------------------------------------------------------------------------
def entrainer(X, a):
    dev = "cuda"
    torch.manual_seed(U.SEED); np.random.seed(U.SEED)
    Xg = torch.from_numpy(X).to(dev)                         # 1,49 M x F en float32 (~1,3 Go) sur le GPU
    N, F = Xg.shape
    model = DAE(F).to(dev)
    ep_depart = 0
    if a.reprise:                                              # reprise depuis un checkpoint existant
        ck = torch.load(DAE_DIR / a.reprise, map_location=dev)
        model.load_state_dict(ck["etat"]); ep_depart = ck["epoques"]
        print(f"[4] reprise depuis {a.reprise} ({ep_depart} époques déjà faites)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    controle = Xg[torch.randperm(N, device=dev)[:20000]]       # échantillon fixe pour suivre la reconstruction
    budget, t0, ep = a.budget_min * 60, time.time(), 0
    print(f"[4] DAE {F}-1500-1500-1500-{F} | {sum(p.numel() for p in model.parameters())/1e6:.1f} M paramètres "
          f"| batch {a.batch} | swap {a.swap} | budget {a.budget_min} min", flush=True)
    while ep < a.max_epochs and time.time() - t0 < budget:
        te = time.time(); model.train()
        perm, somme, nb = torch.randperm(N, device=dev), 0.0, 0
        for i in range(0, N, a.batch):
            xb = Xg[perm[i:i + a.batch]]
            perte = nn.functional.mse_loss(model(swap_noise(xb, Xg, a.swap)), xb)
            opt.zero_grad(set_to_none=True); perte.backward(); opt.step()
            somme += perte.item() if (nb % 50 == 0) else 0.0; nb += 1
        ep += 1
        frac = min(1.0, (time.time() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = a.lr * a.lr_fin ** frac
        model.eval()
        with torch.no_grad():
            mse_propre = nn.functional.mse_loss(model(controle), controle).item()
            mse_bruit = nn.functional.mse_loss(model(swap_noise(controle, Xg, a.swap)), controle).item()
        print(f"ep {ep:3d} | {time.time()-te:5.1f}s | perte entr. ~{somme/max(1, (nb+49)//50):.5f} | "
              f"MSE contrôle propre {mse_propre:.5f} bruité {mse_bruit:.5f} | lr {opt.param_groups[0]['lr']:.2e} "
              f"| écoulé {(time.time()-t0)/60:.1f} min", flush=True)
        if ep % 10 == 0:
            sauver(model, F, ep_depart + ep, a)
    sauver(model, F, ep_depart + ep, a)
    print(f"[4] terminé : {ep} époques (total {ep_depart + ep}) en {(time.time()-t0)/60:.1f} min", flush=True)


def sauver(model, F, ep, a):
    torch.save(dict(etat=model.state_dict(), n_in=F, epoques=ep, batch=a.batch, swap=a.swap, lr=a.lr),
               DAE_DIR / a.sortie)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-min", type=float, default=35)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-fin", type=float, default=0.05, help="facteur du lr atteint à la fin du budget")
    ap.add_argument("--swap", type=float, default=0.15)
    ap.add_argument("--reprise", default="", help="checkpoint de départ dans usine/data/dae (ex. dae.pt)")
    ap.add_argument("--sortie", default="dae.pt", help="nom du checkpoint écrit (ne pas écraser dae.pt en reprise)")
    a = ap.parse_args()
    assert a.sortie not in ("dae_E33f_596ep.pt",) and not (a.reprise and a.sortie == a.reprise), "la reprise doit écrire sous un autre nom"
    # En reprise, on relit les entrées déjà préparées (identiques) pour garantir le même ordre de colonnes
    # Les entrées (déterministes) déjà préparées sont relues : un 2e DAE ou une reprise utilise EXACTEMENT
    # les mêmes colonnes que le DAE d'origine (et 13_nn_dae.py lit ce même fichier).
    X = np.load(DAE_DIR / "X_all.npy") if (DAE_DIR / "X_all.npy").exists() else preparer_entrees()
    entrainer(X, a)
