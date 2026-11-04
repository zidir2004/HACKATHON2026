"""15_verif_test_E32.py — contrôle de cohérence des prédictions TEST du réseau (piste B) :
la corrélation de rang réseau/boostings sur le test doit ressembler à celle observée sur l'OOF."""
import pathlib, sys
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import utils as U  # noqa: E402
L = lambda e, s: pd.read_csv(U.OOF_DIR / f"{e}_{s}.csv").sort_values(U.ID, ignore_index=True)
for e in ["E32b", "E32c"]:
    print(e, "Spearman vs E09 : OOF", round(spearmanr(L(e, "oof")["oof"], L("E09", "oof")["oof"]).correlation, 4),
          "| test", round(spearmanr(L(e, "test")["prediction"], L("E09", "test")["prediction"]).correlation, 4))
t = L("E32b", "test")["prediction"]; o = L("E32b", "oof")["oof"]
print("E32b moyenne proba : OOF", round(o.mean(), 4), "test", round(t.mean(), 4))
print("Spearman 2_306 vs 2_304 :", round(spearmanr(pd.read_csv(U.SUB_DIR / "2_306.csv", header=None)[1],
                                                pd.read_csv(U.SUB_DIR / "2_304.csv", header=None)[1]).correlation, 4))
