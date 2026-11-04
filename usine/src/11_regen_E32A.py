"""11_regen_E32A.py — régénère les fichiers OOF/test de E32A (piste A : LightGBM Jahrer + one-hot E30b,
moyenne des probabilités de 3 graines 42/7/2026) SOUS SON NOM, sans toucher à oof/E32_* (réseau piste B)
ni au registre. Rend reproductible 2_305.csv :
    python -u usine/src/11_regen_E32A.py
    python -u usine/src/07_rangs_vers_proba.py --exp E34 --membres E32A E07 E08 --sortie usine/submissions/2_305.csv
"""
import pathlib, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402

runs = ["E30b", "E30bs7", "E30bs2026"]
oofs = [pd.read_csv(U.OOF_DIR / f"{r}_oof.csv").sort_values(U.ID, ignore_index=True) for r in runs]
tests = [pd.read_csv(U.OOF_DIR / f"{r}_test.csv").sort_values(U.ID, ignore_index=True) for r in runs]
y, fold = oofs[0][U.TARGET].to_numpy(), oofs[0]["fold"].to_numpy()
p = np.mean([o["oof"] for o in oofs], axis=0)
gs = [U.gini(y[fold == k], p[fold == k]) for k in range(U.N_FOLDS)]
pd.DataFrame({U.ID: oofs[0][U.ID], U.TARGET: y, "oof": p, "fold": fold}).to_csv(U.OOF_DIR / "E32A_oof.csv", index=False)
pd.DataFrame({U.ID: tests[0][U.ID], "prediction": np.mean([t["prediction"] for t in tests], axis=0)}).to_csv(
    U.OOF_DIR / "E32A_test.csv", index=False)
pd.DataFrame({"n_arbres": "n/a", "fold": range(5), "gini": gs, "duree_s": 0}).to_csv(
    U.OOF_DIR / "E32A_folds_detail.csv", index=False)
print(f"E32A : gini CV {np.mean(gs):.5f} ± {np.std(gs):.5f} | OOF {U.gini(y, p):.5f} (attendu 0.28052 / 0.28039)")
