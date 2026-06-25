# 11_prc_seuil_evt.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)
df["total_affected"] = pd.to_numeric(df["total_affected"], errors="coerce")

# Variable de severite : total_affected au niveau evenement
evt = df.groupby("group_uuid")["total_affected"].max()
x = evt.dropna()
x = x[x > 1].sort_values().values
print(f"Evenements exploitables : {len(x):,}")
print(f"Min / Med / Max : {x.min():,.0f} / {np.median(x):,.0f} / {x.max():,.0f}")

# --- 1. Mean Excess Plot ---
# e(u) = moyenne des (x - u) pour x > u. Linéaire croissant => queue type GPD.
seuils = np.quantile(x, np.linspace(0.5, 0.995, 60))
me = [ (x[x > u] - u).mean() for u in seuils ]
n_exc = [ (x > u).sum() for u in seuils ]

# --- 2. Stabilite des parametres GPD selon le seuil ---
xis, sigmas, us = [], [], []
for u in np.quantile(x, np.linspace(0.80, 0.99, 25)):
    exces = x[x > u] - u
    if len(exces) > 50:
        xi, loc, sigma = genpareto.fit(exces, floc=0)
        xis.append(xi); sigmas.append(sigma); us.append(u)

# --- Graphiques ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(seuils, me, marker=".")
axes[0].set_xscale("log"); axes[0].set_title("Mean Excess Plot")
axes[0].set_xlabel("seuil u (log)"); axes[0].set_ylabel("exces moyen e(u)")

axes[1].plot(us, xis, marker="o", color="darkred")
axes[1].set_xscale("log"); axes[1].set_title("Stabilite de xi (forme)")
axes[1].set_xlabel("seuil u (log)"); axes[1].set_ylabel("xi")
axes[1].axhspan(1.1, 1.5, alpha=0.15, color="green")  # ta plage cible xi in [1.1,1.5]

axes[2].plot(us, sigmas, marker="o", color="navy")
axes[2].set_xscale("log"); axes[2].set_title("Stabilite de sigma (echelle)")
axes[2].set_xlabel("seuil u (log)"); axes[2].set_ylabel("sigma")

plt.tight_layout()
out = r"C:\Users\KélianKADDOURI\projet-dora\briques\evt_diagnostics.png"
plt.savefig(out, dpi=120)
print(f"\nGraphiques sauvegardes : {out}")

# Table indicative : seuil / nb exces / xi
print("\n=== Seuil candidat / nb exces / xi ===")
for u, xi in zip(us, xis):
    print(f"  u={u:>14,.0f}   n_exces={int((x>u).sum()):>5}   xi={xi:.3f}")

import pandas as pd
import numpy as np

df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)
df["total_affected"] = pd.to_numeric(df["total_affected"], errors="coerce")
evt = df.groupby("group_uuid")["total_affected"].max().dropna()

top = evt.sort_values(ascending=False).head(15)
print("=== 15 plus grands evenements ===")
print(top.to_string())

# Combien de valeurs "rondes" suspectes (exactement 1e9, 2e9...) ?
for v in [1_000_000_000, 2_000_000_000, 500_000_000, 100_000_000]:
    print(f"  == {v:>15,} : {(evt == v).sum()} evenements")