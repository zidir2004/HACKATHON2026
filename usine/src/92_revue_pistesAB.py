"""92_revue_pistesAB.py - controle reviewer (n'ecrit NI registre NI oof). numpy+pandas seulement.
Format de 2_305/2_306, reconstruction de E32A/E34/E32c/E33f depuis les membres, Gini par fold,
deltas apparies E33f vs E34 vs E09, bootstrap de lignes intra-fold du delta.
Usage : python -u usine/src/92_revue_pistesAB.py"""
import pathlib, sys, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[2]; O = R / "usine/experiences/oof"; S = R / "usine/submissions"
def rk(x): return pd.Series(x).rank(method="average").to_numpy()
def auc(y, p):
    r = rk(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def gini(y, p): return 2 * auc(y, p) - 1
def L(e, s):
    d = pd.read_csv(O / f"{e}_{s}.csv").sort_values("id", ignore_index=True); return d
def sp(a, b): return np.corrcoef(rk(a), rk(b))[0, 1]
# ---- 1. format brut
ids_test = pd.read_csv(R / "test.csv", usecols=["id"])["id"].to_numpy()
subs = {}
for f in ["2_305", "2_306"]:
    raw = (S / f"{f}.csv").read_bytes()
    lignes = raw.split(b"\n"); fin_lf = raw.endswith(b"\n"); lignes = [l for l in lignes if l]
    crlf = b"\r" in raw
    ok_cols = all(l.count(b",") == 1 for l in lignes)
    sub = pd.read_csv(S / f"{f}.csv", header=None, names=["id", "p"])
    subs[f] = sub
    print(f"[1] {f}: {len(lignes)} lignes, 1re ligne {lignes[0][:40]!r}, CR={crlf}, LF final={fin_lf}, 2 col partout={ok_cols}, "
          f"ids==test.csv meme ordre={len(sub)==len(ids_test) and (sub.id.to_numpy()==ids_test).all()}, "
          f"doublons={sub.id.duplicated().sum()}, NaN={sub.p.isna().sum()}, min={sub.p.min():.5f} max={sub.p.max():.5f} "
          f"moy={sub.p.mean():.5f}, dtype={sub.p.dtype}, nb valeurs distinctes={sub.p.nunique()}")
tr_mean = pd.read_csv(R / "train.csv", usecols=["target"])["target"].mean(); print(f"    taux positif train {tr_mean:.5f}")
# ---- 2. reconstruction
o = {e: L(e, "oof") for e in ["E09", "E34", "E33f", "E30", "E30b", "E30bs7", "E30bs2026", "E07", "E08",
                                "E32b", "E32b_g1", "E32b_g2", "E32c", "E31c", "E33e", "E02"]}
t = {e: L(e, "test") for e in ["E34", "E33f", "E30", "E30b", "E30bs7", "E30bs2026", "E07", "E08", "E32b", "E32b_g1", "E32b_g2", "E32c"]}
y = o["E09"]["target"].to_numpy(); fold = o["E09"]["fold"].to_numpy(); ids = o["E09"]["id"].to_numpy()
for e, d in o.items():
    assert (d.id.to_numpy() == ids).all() and (d.fold.to_numpy() == fold).all() and (d.target.to_numpy() == y).all(), e
for e, d in t.items(): assert (d.id.to_numpy() == ids_test).all(), e
print("[2] ids/folds/cible identiques pour tous les OOF ; ids test identiques")
N, Nt = len(y), len(ids_test)
def blend(mem, src, col): return np.mean([rk(src[m][col].to_numpy()) / len(src[m]) for m in mem], axis=0)
e32A_oof = np.mean([o[m]["oof"] for m in ["E30b", "E30bs7", "E30bs2026"]], axis=0)
e32A_te = np.mean([t[m]["prediction"] for m in ["E30b", "E30bs7", "E30bs2026"]], axis=0)
o["E32A"] = pd.DataFrame({"oof": e32A_oof}); t["E32A"] = pd.DataFrame({"prediction": e32A_te})
for nom, mem in [("E34", ["E32A", "E07", "E08"]), ("E32c", ["E32b", "E32b_g1", "E32b_g2"]), ("E33f", ["E30", "E07", "E08", "E32c"])]:
    bo, bt = blend(mem, o, "oof"), blend(mem, t, "prediction")
    print(f"[2] {nom} recalcule depuis {mem}: max|dOOF|={np.abs(bo-o[nom]['oof']).max():.2e} max|dTEST|={np.abs(bt-t[nom]['prediction']).max():.2e}")
# quantile mapping
def qmap(mel, mem):
    mp = np.mean([t[m]["prediction"].to_numpy() for m in mem], axis=0); ordre = np.argsort(mel, kind="mergesort")
    p = np.empty(len(mel)); p[ordre] = np.sort(mp); return p
for f, nom, mem in [("2_305", "E34", ["E32A", "E07", "E08"]), ("2_306", "E33f", ["E30", "E07", "E08", "E32b"])]:
    p = qmap(t[nom]["prediction"].to_numpy(), mem); s = subs[f].p.to_numpy()
    print(f"[2] {f}: max|ecart| vs recalcul={np.abs(p-s).max():.2e} | Spearman vs {nom}_test={sp(s, t[nom]['prediction']):.6f} "
          f"| ex-aequo melange test={Nt-t[nom]['prediction'].nunique()}")
# fichier NN actuel E32 (piste B) aurait-il servi ? corr 2_305 vs E32-NN mapping
nn = L("E32", "test")["prediction"].to_numpy()
p_bad = qmap(t["E34"]["prediction"].to_numpy(), ["E07", "E08"]); 
print(f"[2] controle ecrasement: moyenne test E32 actuel (NN)={nn.mean():.5f}, E32A recalcule={e32A_te.mean():.5f}")
print(f"[2] Spearman 2_305 vs 2_306 = {sp(subs['2_305'].p, subs['2_306'].p):.5f}")
# ---- 3. gini par fold
def gf(p): return np.array([gini(y[fold == k], p[fold == k]) for k in range(5)])
G = {e: gf(o[e]["oof"].to_numpy()) for e in ["E09", "E34", "E33f", "E33e", "E31c", "E32c", "E32b", "E30", "E32A"]}
for e, g in G.items(): print(f"[3] {e:5s} folds {np.round(g,5).tolist()} moy {g.mean():.5f} sd {g.std():.5f} | OOF {gini(y, o[e]['oof'].to_numpy()):.5f}")
for a_, b_ in [("E33f", "E09"), ("E34", "E09"), ("E33f", "E34"), ("E33f", "E31c"), ("E33f", "E33e")]:
    d = G[a_] - G[b_]; se = d.std(ddof=1) / np.sqrt(5)
    print(f"[3] {a_} - {b_}: d/fold {np.round(d,5).tolist()} moy {d.mean():+.5f} se {se:.5f} gagnes {(d>0).sum()}/5")
cols = ["E30", "E32A", "E07", "E08", "E32c", "E32b"]
C = pd.DataFrame({c: rk(o[c]["oof"].to_numpy()) for c in cols}).corr().round(3); print("[3] Spearman OOF\n", C)
# ---- 4. bootstrap lignes intra-fold du delta OOF-gini E33f-E34 et E33f-E09
rng = np.random.default_rng(92); B = 60
pa, pb, pc = o["E33f"]["oof"].to_numpy(), o["E34"]["oof"].to_numpy(), o["E09"]["oof"].to_numpy()
idx_f = [np.where(fold == k)[0] for k in range(5)]
d1, d2 = [], []
for b in range(B):
    ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_f])
    ga, gb, gc = (np.mean([gini(y[ii][fold[ii]==k], p[ii][fold[ii]==k]) for k in range(5)]) for p in (pa, pb, pc))
    d1.append(ga - gb); d2.append(ga - gc)
for n, d in [("E33f-E34", d1), ("E33f-E09", d2)]:
    d = np.array(d); print(f"[4] bootstrap {B} {n}: moy {d.mean():+.5f} sd {d.std():.5f} P(d<=0)={np.mean(d<=0):.3f} IC90 [{np.quantile(d,.05):+.5f}, {np.quantile(d,.95):+.5f}]")
