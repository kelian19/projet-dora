# 11b_prc_mrl_hill.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)
df["total_affected"] = pd.to_numeric(df["total_affected"], errors="coerce")
evt = df.groupby("group_uuid")["total_affected"].max().dropna()
x = np.sort(evt[evt > 1].values)
N = len(x)
print(f"N = {N:,}")

# ============ 1. MRL / Mean Excess Plot avec IC ============
# e(u) = moyenne des exces ; IC ~ +/- 1.96 * sd/sqrt(n_u)
seuils = np.quantile(x, np.linspace(0.50, 0.995, 80))
mrl, lo, hi = [], [], []
for u in seuils:
    e = x[x > u] - u
    if len(e) > 5:
        m = e.mean(); se = e.std(ddof=1)/np.sqrt(len(e))
        mrl.append(m); lo.append(m - 1.96*se); hi.append(m + 1.96*se)
    else:
        mrl.append(np.nan); lo.append(np.nan); hi.append(np.nan)

# ============ 2. Hill estimateur ============
# Pour les k plus grandes valeurs : alpha_Hill(k) = 1 / [ (1/k) * sum_{i=1}^{k} ln(x_(N-i+1)) - ln(x_(N-k)) ]
# puis xi = 1/alpha
xs_desc = x[::-1]  # ordre decroissant
def hill_xi(k):
    logs = np.log(xs_desc[:k])
    alpha = 1.0 / (logs.mean() - np.log(xs_desc[k]))
    return 1.0 / alpha   # xi

ks = np.arange(20, 3000)
xis_hill = np.array([hill_xi(k) for k in ks])
# seuil correspondant a chaque k (la k-ieme plus grande valeur)
seuils_k = xs_desc[ks]

# ============ Graphiques ============
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# MRL
axes[0].plot(seuils, mrl, color="navy")
axes[0].fill_between(seuils, lo, hi, alpha=0.2, color="navy")
axes[0].set_xscale("log"); axes[0].set_title("MRL / Mean Excess Plot")
axes[0].set_xlabel("seuil u (log)"); axes[0].set_ylabel("e(u)")

# Hill plot vs k
axes[1].plot(ks, xis_hill, color="darkred")
axes[1].axhspan(1.1, 1.5, alpha=0.15, color="green")
axes[1].set_title("Hill plot (xi = 1/alpha) vs k")
axes[1].set_xlabel("k (nb plus grandes valeurs)"); axes[1].set_ylabel("xi (Hill)")

# Hill plot vs seuil (pour comparer au fit GPD)
axes[2].plot(seuils_k, xis_hill, color="darkgreen")
axes[2].axhspan(1.1, 1.5, alpha=0.15, color="green")
axes[2].axvline(128467, color="orange", ls="--", label="U=128 467")
axes[2].set_xscale("log"); axes[2].set_title("Hill (xi) vs seuil")
axes[2].set_xlabel("seuil (log)"); axes[2].set_ylabel("xi (Hill)")
axes[2].legend()

plt.tight_layout()
out = r"C:\Users\KélianKADDOURI\projet-dora\briques\evt_mrl_hill.png"
plt.savefig(out, dpi=120)
print(f"Graphiques sauvegardes : {out}")

# Table Hill autour du seuil retenu
print("\n=== Hill xi pour quelques k ===")
for k in [50, 100, 200, 500, 1000, 1358, 2000]:
    if k < len(xs_desc):
        print(f"  k={k:>5}  seuil={xs_desc[k]:>14,.0f}  xi_Hill={hill_xi(k):.3f}")