"""93_revue_finale.py - controle reviewer final (n'ecrit NI registre NI oof). numpy+pandas.
Format 2_306/2_307, reconstruction E51c/E56r/E56, Gini par fold, deltas E56 vs E33f, bootstrap lignes
intra-fold, controle par graine du reseau E51, Spearman test 2_306/2_307.
Usage : python -u usine/src/93_revue_finale.py"""
import pathlib, numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[2]; O = R / "usine/experiences/oof"; S = R / "usine/submissions"
def rk(x): return pd.Series(x).rank(method="average").to_numpy()
def auc(y, p):
    r = rk(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def gini(y, p): return 2 * auc(y, p) - 1
def L(e, s): return pd.read_csv(O / f"{e}_{s}.csv").sort_values("id", ignore_index=True)
def sp(a, b): return np.corrcoef(rk(a), rk(b))[0, 1]
ids_test = pd.read_csv(R / "test.csv", usecols=["id"])["id"].to_numpy()
subs = {}
for f in ["2_306", "2_307"]:
    raw = (S / f"{f}.csv").read_bytes(); cr, lf = b"\r" in raw, raw.endswith(b"\n"); lignes = [l for l in raw.split(b"\n") if l]
    c2 = all(l.count(b",") == 1 for l in lignes)
    sub = pd.read_csv(S / f"{f}.csv", header=None, names=["id", "p"]); subs[f] = sub
    print(f"[1] {f}: {len(lignes)} lignes, 1re {lignes[0][:40]!r}, CR={cr}, LF final={lf}, "
          f"2 col={c2}, ids==test ordre={len(sub)==len(ids_test) and (sub.id.to_numpy()==ids_test).all()}, "
          f"doublons={sub.id.duplicated().sum()}, NaN={sub.p.isna().sum()}, min={sub.p.min():.5f} max={sub.p.max():.5f} moy={sub.p.mean():.5f} dtype={sub.p.dtype}")
noms_o = ["E09", "E33f", "E30", "E07", "E08", "E32b", "E32c", "E51_g0", "E51_g1", "E51_g2", "E51c", "E56r", "E56", "E55", "E59", "E53", "E54", "E57", "E58", "E32b_g1", "E32b_g2"]
o = {e: L(e, "oof") for e in noms_o}
t = {e: L(e, "test") for e in ["E33f", "E30", "E07", "E08", "E32b", "E32c", "E51_g0", "E51_g1", "E51_g2", "E51c", "E56r", "E56"]}
y = o["E09"]["target"].to_numpy(); fold = o["E09"]["fold"].to_numpy(); ids = o["E09"]["id"].to_numpy()
for e, d in o.items(): assert (d.id.to_numpy()==ids).all() and (d.fold.to_numpy()==fold).all() and (d.target.to_numpy()==y).all(), e
for e, d in t.items(): assert (d.id.to_numpy()==ids_test).all(), e
print("[2] ids/folds/cible identiques ; ids test identiques")
def blend(mem, src, col): return np.mean([rk(src[m][col].to_numpy()) / len(src[m]) for m in mem], axis=0)
for nom, mem in [("E51c", ["E51_g0","E51_g1","E51_g2"]), ("E56r", ["E32c","E51c"]), ("E56", ["E30","E07","E08","E56r"])]:
    bo, bt = blend(mem, o, "oof"), blend(mem, t, "prediction")
    print(f"[2] {nom} recalcule {mem}: max|dOOF|={np.abs(bo-o[nom]['oof']).max():.2e} max|dTEST|={np.abs(bt-t[nom]['prediction']).max():.2e}")
mp = np.mean([t[m]["prediction"].to_numpy() for m in ["E30","E07","E08","E32b","E51_g0"]], axis=0)
mel = t["E56"]["prediction"].to_numpy(); ordre = np.argsort(mel, kind="mergesort"); p = np.empty(len(mel)); p[ordre] = np.sort(mp)
s7 = subs["2_307"].p.to_numpy(); s6 = subs["2_306"].p.to_numpy()
print(f"[2] 2_307 max|ecart| recalcul={np.abs(p-s7).max():.2e} Spearman vs E56_test={sp(s7, mel):.6f} ex-aequo={len(mel)-pd.Series(mel).nunique()}")
print(f"[2] Spearman test 2_306 vs 2_307 = {sp(s6, s7):.5f} ; Spearman 2_306 vs E33f_test={sp(s6, t['E33f']['prediction']):.6f}")
r6, r7 = rk(s6)/len(s6), rk(s7)/len(s7); top = lambda r: r > 1 - 0.037
print(f"[2] |drang| moy {np.abs(r6-r7).mean():.5f} p99 {np.quantile(np.abs(r6-r7),.99):.5f} ; recouvrement top 3,7 % = {(top(r6)&top(r7)).sum()/top(r6).sum():.4f}")
def gf(p): return np.array([gini(y[fold==k], p[fold==k]) for k in range(5)])
G = {e: gf(o[e]["oof"].to_numpy()) for e in noms_o}
for e in ["E09","E33f","E56","E55","E59","E32c","E51c","E51_g0","E51_g1","E51_g2"]:
    print(f"[3] {e:6s} folds {np.round(G[e],5).tolist()} moy {G[e].mean():.5f} sd {G[e].std():.5f}")
for a_ in ["E56","E55","E59","E53","E54","E57","E58"]:
    d = G[a_]-G["E33f"]; print(f"[3] {a_}-E33f d/fold {np.round(d,5).tolist()} moy {d.mean():+.5f} se {d.std(ddof=1)/np.sqrt(5):.5f} {(d>0).sum()}/5")
# robustesse par graine : E56 avec une seule graine E51 (et E32 correspondante)
for g in ["0","1","2"]:
    e32 = "E32b" if g=="0" else f"E32b_g{g}"
    r = np.mean([rk(o[e32]["oof"].to_numpy()), rk(o[f"E51_g{g}"]["oof"].to_numpy())], axis=0)
    o[f"r{g}"] = pd.DataFrame({"oof": r})
    v56 = blend(["E30","E07","E08",f"r{g}"], o, "oof"); v33 = blend(["E30","E07","E08",e32], o, "oof")
    d = gf(v56)-gf(v33); print(f"[3b] graine {g}: E56(g)-E33f(g) d/fold {np.round(d,5).tolist()} moy {d.mean():+.5f} {(d>0).sum()}/5")
# bootstrap lignes intra-fold
rng = np.random.default_rng(93); B = 50; idx_f = [np.where(fold==k)[0] for k in range(5)]
cand = ["E33f","E55","E56","E59"]; P = {c: o[c]["oof"].to_numpy() for c in cand}
res = []
for b in range(B):
    ii = [rng.choice(ix, len(ix)) for ix in idx_f]
    res.append({c: np.mean([gini(y[i], P[c][i]) for i in ii]) for c in cand})
res = pd.DataFrame(res); d = res["E56"]-res["E33f"]
print(f"[4] bootstrap {B} E56-E33f: moy {d.mean():+.5f} sd {d.std():.5f} P(d<=0)={np.mean(d<=0):.3f} IC90 [{d.quantile(.05):+.5f}, {d.quantile(.95):+.5f}]")
print(f"[4] frequence du meilleur parmi {cand}: {res.idxmax(axis=1).value_counts().to_dict()}")
# bootstrap des folds
fb = [rng.integers(0,5,5) for _ in range(2000)]; dd = G["E56"]-G["E33f"]
print(f"[4] bootstrap folds: P(moy d<=0)={np.mean([dd[i].mean()<=0 for i in fb]):.3f}")
