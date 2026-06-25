# 13_prc_jacobs_conversion_corrigee.py
import numpy as np
import pandas as pd
from scipy.stats import genpareto
import json
import os

ROOT = r"C:\Users\KélianKADDOURI\projet-dora\briques"

# --- 1. Charger les parametres GPD valides ---
with open(os.path.join(ROOT, "params_gpd.json")) as f:
    gpd = json.load(f)

U   = gpd["seuil_U"]
xi  = gpd["xi_central_retenu"]      # 1.30
sig = gpd["sigma"]
p_dep = gpd["prob_depassement"]

# --- 2. Reconstruire la distribution de volume ---
df = pd.read_csv(os.path.join(ROOT, "Data_Breach_Chronology.csv"), sep="|", 
                 encoding="utf-8-sig", low_memory=False)
df["total_affected"] = pd.to_numeric(df["total_affected"], errors="coerce")
x = df.groupby("group_uuid")["total_affected"].max().dropna()
x = np.sort(x[x > 1].values)

def quantile_volume(p):
    if p <= 1 - p_dep:
        return np.quantile(x, p)
    p_cond = (p - (1 - p_dep)) / p_dep
    return U + genpareto.ppf(p_cond, xi, loc=0, scale=sig)

def simuler_volumes(M=1_000_000, seed=42):
    """Simule M volumes pour calculer la TVaR empirique."""
    rng = np.random.default_rng(seed)
    u_rand = rng.random(M)
    vol = np.empty(M)
    mask_corps = u_rand <= (1 - p_dep)
    # Tirage empirique pour le corps
    vol[mask_corps] = rng.choice(x[x <= U], size=mask_corps.sum(), replace=True)
    # Tirage GPD pour la queue
    vol[~mask_corps] = U + genpareto.rvs(xi, loc=0, scale=sig, size=(~mask_corps).sum(), random_state=rng)
    return vol

# ============ 3. PARAMETRES JACOBS ET CAPS ============
USD_EUR = 0.92        
A_CENTRAL = 7.68      # Intercept fixe pour rester dans la bonne echelle

SCENARIOS_B = {
    "Pente basse (0.65)": 0.65,
    "Central (0.76)": 0.76,
    "Pente haute (0.85)": 0.85
}

CAPS = {
    "Cap 40M€": 40_000_000,
    "Cap 100M€": 100_000_000,
    "Sans Cap": np.inf
}

def cout_eur(volume, b, cap):
    """Volume (personnes) -> cout EUR via Jacobs log-log, avec cap."""
    volume = np.maximum(volume, 1)
    cout_usd = np.exp(A_CENTRAL + b * np.log(volume))
    return np.minimum(cout_usd * USD_EUR, cap)

# ============ 4. EXAMEN TVaR 99% (Moyenne de la queue) ============
print("Simulation d'un million d'incidents pour évaluer la TVaR 99%...")
M_SIM = 1_000_000
vol_sim = simuler_volumes(M_SIM)

print("\n=== SENSIBILITÉ DE LA TVaR 99% (en Millions €) ===")
print(f"{'Scénario Pente':<20} | {'Cap 40M€':<10} | {'Cap 100M€':<10} | {'Sans Cap':<15} | xi_effectif")
print("-" * 75)

for nom_b, b in SCENARIOS_B.items():
    tvars = []
    xi_eff = xi * b
    for nom_cap, cap in CAPS.items():
        couts = cout_eur(vol_sim, b, cap)
        # --- CORRECTION ICI ---
        # On trie et on prend exactement les 1% pires, même si le plafond est atteint.
        couts_sorted = np.sort(couts)
        tvar_99 = couts_sorted[int(0.99 * M_SIM):].mean()
        tvars.append(tvar_99 / 1e6)
        
    inf_marker = "(! EXPLOSE !)" if xi_eff >= 1 else ""
    print(f"{nom_b:<20} | {tvars[0]:>10.1f} | {tvars[1]:>10.1f} | {tvars[2]:>10.1f} {inf_marker:<13} | {xi_eff:.3f}")

# ============ 5. QUANTILES (VaR) SCENARIO CENTRAL ============
print("\n=== QUANTILES DE SEVERITE INDIVIDUELLE : SCENARIO CENTRAL (b=0.76) ===")
ps = [0.50, 0.90, 0.95, 0.99, 0.995, 0.999]
print(f"{'p':>8} {'Volume (pers.)':>16} {'Cap 40M€':>16} {'Cap 100M€':>16} {'Sans Cap':>16}")
for p in ps:
    v = quantile_volume(p)
    c_40  = cout_eur(v, SCENARIOS_B["Central (0.76)"], CAPS["Cap 40M€"])
    c_100 = cout_eur(v, SCENARIOS_B["Central (0.76)"], CAPS["Cap 100M€"])
    c_inf = cout_eur(v, SCENARIOS_B["Central (0.76)"], CAPS["Sans Cap"])
    print(f"{p*100:>7.2f}% {v:>16,.0f} {c_40:>16,.0f} {c_100:>16,.0f} {c_inf:>16,.0f}")