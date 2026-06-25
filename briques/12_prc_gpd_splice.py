# 12_prc_gpd_splice.py
import pandas as pd
import numpy as np
from scipy.stats import genpareto
import json

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)
df["total_affected"] = pd.to_numeric(df["total_affected"], errors="coerce")
evt = df.groupby("group_uuid")["total_affected"].max().dropna()
x = evt[evt > 1].sort_values().values
N = len(x)

U = 128_467   # seuil retenu (creux de stabilite)
exces = x[x > U] - U
n_exc = len(exces)
prob_depassement = n_exc / N

print(f"N total       : {N:,}")
print(f"Seuil U       : {U:,}")
print(f"Nb exces      : {n_exc:,}")
print(f"P(X > U) emp. : {prob_depassement:.4f}")

# --- Fit GPD libre (xi estime) ---
xi, loc, sigma = genpareto.fit(exces, floc=0)
print(f"\n=== GPD fit libre ===")
print(f"xi (forme)      : {xi:.4f}")
print(f"sigma (echelle) : {sigma:,.0f}")

# --- IC bootstrap sur xi et sigma ---
B = 1000
rng = np.random.default_rng(42)
xis, sigmas = [], []
for _ in range(B):
    ech = rng.choice(exces, size=n_exc, replace=True)
    try:
        xb, _, sb = genpareto.fit(ech, floc=0)
        xis.append(xb); sigmas.append(sb)
    except Exception:
        pass
xis, sigmas = np.array(xis), np.array(sigmas)
print(f"\nIC95% xi    : [{np.percentile(xis,2.5):.3f}, {np.percentile(xis,97.5):.3f}]")
print(f"IC95% sigma : [{np.percentile(sigmas,2.5):,.0f}, {np.percentile(sigmas,97.5):,.0f}]")

# --- Quantiles spliced (volume de personnes, AVANT Jacobs) ---
def quantile_spliced(p, xi_use, sigma_use):
    if p <= 1 - prob_depassement:
        return np.quantile(x, p)
    p_cond = (p - (1 - prob_depassement)) / prob_depassement
    return U + genpareto.ppf(p_cond, xi_use, loc=0, scale=sigma_use)

print(f"\n=== Quantiles de severite (VOLUME personnes) ===")
print(f"{'p':>8} {'xi=fit':>18} {'xi=1.30':>18}")
for p in [0.99, 0.995, 0.999, 0.9995]:
    q_fit = quantile_spliced(p, xi, sigma)
    q_130 = quantile_spliced(p, 1.30, sigma)
    print(f"{p*100:>7.2f}% {q_fit:>18,.0f} {q_130:>18,.0f}")

# --- Sauvegarde parametres ---
params = {
    "variable": "total_affected (volume personnes, niveau evenement)",
    "seuil_U": U,
    "xi_fit": float(xi),
    "xi_central_retenu": 1.30,
    "xi_plage_sensibilite": [1.25, 1.55],
    "sigma": float(sigma),
    "prob_depassement": float(prob_depassement),
    "N": int(N), "n_exces": int(n_exc),
    "xi_ic95": [float(np.percentile(xis,2.5)), float(np.percentile(xis,97.5))],
    "sigma_ic95": [float(np.percentile(sigmas,2.5)), float(np.percentile(sigmas,97.5))],
}
out = r"C:\Users\KélianKADDOURI\projet-dora\briques\params_gpd.json"
with open(out, "w") as f:
    json.dump(params, f, indent=2)
print(f"\nParametres sauvegardes : {out}")