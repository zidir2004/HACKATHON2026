"""01_audit_extra.py - sondes complementaires (lecture seule) : ps_ind_14, structure ps_reg_03 / ps_car_13, doublons hors calc."""
import json
import numpy as np, pandas as pd
tr = pd.read_csv("train.csv", low_memory=False)
s = tr[[f"ps_ind_{i}_bin" for i in (10, 11, 12, 13)]].sum(axis=1)
print("ps_ind_14 == somme(ps_ind_10..13_bin) :", float((s == tr["ps_ind_14"]).mean()))
print(pd.crosstab(tr["ps_ind_14"], s))
for c in ["ps_reg_03", "ps_car_13", "ps_car_12", "ps_car_14"]:
    u = np.unique(tr[c][tr[c] != -1].to_numpy())
    for k in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
        x = u * u * k
        tol = 1e-3 * max(1.0, k / 1e4)
        print(c, "carre*%g entier (tol %.0e):" % (k, tol), round(float((np.abs(x - np.round(x)) < tol).mean()), 4), "| chance ~", round(2 * tol, 4))
# ps_car_13 : exemples de ps_car_13^2 * 1000 pour voir une structure
u = np.unique(tr["ps_car_13"])
print("ps_car_13 premiers carres:", [round(float(x * x), 6) for x in u[:8]])
u = np.unique(tr["ps_reg_03"][tr["ps_reg_03"] != -1])
print("ps_reg_03 premiers carres:", [round(float(x * x), 6) for x in u[:12]])
# doublons hors calc
F = [c for c in tr.columns if c not in ("id", "target") and "calc" not in c]
d = tr[tr.duplicated(F, keep=False)].sort_values(F)
print("lignes impliquees dans des doublons hors calc:", len(d), "| cibles:", d["target"].tolist())
print("manquants par bloc :")
mv = ["ps_ind_02_cat", "ps_ind_04_cat", "ps_ind_05_cat", "ps_car_01_cat", "ps_car_02_cat", "ps_car_03_cat", "ps_car_05_cat", "ps_car_07_cat", "ps_car_09_cat", "ps_reg_03", "ps_car_14"]
print((tr[mv] == -1).astype(int).corr().round(2).to_string())
r = json.load(open("usine/rapports/audit_resume.json", encoding="utf-8"))
for k in ["doublons_hors_calc_lignes_en_trop_train", "doublons_hors_calc_taille_max_groupe", "lignes_test_identiques_train_hors_calc",
          "binaire_deduite_des_autres_du_groupe", "blocs_manquants_corr>0.3", "ecart_max_float32", "mem_train_typee_Mo", "mem_train_brute_Mo",
          "derive_psi_max", "derive_ks_max", "derive_ecart_max_pct_manq", "corr_cible_top", "paires_corr>0.9"]:
    print(k, "=", r.get(k))
