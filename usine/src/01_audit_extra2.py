"""Sonde : ps_reg_03^2 * 1600 est-il entier ? ps_car_13^2 * k ?"""
import numpy as np, pandas as pd
tr = pd.read_csv("train.csv", usecols=["ps_reg_03", "ps_car_13"], low_memory=False)
for c in ["ps_reg_03", "ps_car_13"]:
    u = np.unique(tr[c][tr[c] != -1].to_numpy())
    for k in (1600, 800, 400, 6400, 3200):
        x = u * u * k
        print(c, k, "frac entiers (tol 2e-3):", round(float((np.abs(x - np.round(x)) < 2e-3).mean()), 4))
