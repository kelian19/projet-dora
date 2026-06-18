# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITE AU CHOIX DE LOI — BRIQUE SANCTION
=============================================================================
Montre que la VaR de la brique sanction est peu sensible au choix de la FORME 
de loi, justifiant que le choix de la Bêta n'est pas un point de fragilité.
"""
import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42
PROBA_SURVENANCE = 0.10

def ratio_beta(rng, n, alpha=0.8, beta=6.0): return rng.beta(alpha, beta, size=n)
def ratio_uniforme(rng, n, borne_max=0.5): return rng.uniform(0.0, borne_max, size=n)
def ratio_triangulaire(rng, n, mode=0.05, borne_max=0.6): return rng.triangular(0.0, mode, borne_max, size=n)
def ratio_kumaraswamy(rng, n, a=1.2, b=12.0):
    u = rng.random(n)
    return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)
def ratio_pert(rng, n, mini=0.0, mode=0.05, maxi=0.6, lamb=4.0):
    alpha = 1.0 + lamb * (mode - mini) / (maxi - mini)
    beta = 1.0 + lamb * (maxi - mode) / (maxi - mini)
    return mini + rng.beta(alpha, beta, size=n) * (maxi - mini)

def simuler_sanction_loi(loi_ratio, plafond, M, rng, proba=PROBA_SURVENANCE, **kw):
    survient = rng.random(M) < proba
    ratios = loi_ratio(rng, M, **kw)
    return np.where(survient, ratios * plafond, 0.0)

def main():
    plafond = 800e6 * 0.02 # 16 M€
    mu_cible = 0.8 / (0.8 + 6.0)

    print("=" * 74)
    print(" SENSIBILITE AU CHOIX DE LOI — BRIQUE SANCTION")
    print("=" * 74)
    
    m2 = 2 * mu_cible
    tri_max = 3 * mu_cible / 1.4 
    tri_mode = 0.4 * tri_max
    pert_max = 0.45 
    pert_mode = (6 * mu_cible - pert_max) / 4.0

    lois = [
        ("Beta(0.8, 6.0) [REF]", ratio_beta, {}),
        ("Uniforme(0, 2mu)",     ratio_uniforme, {"borne_max": m2}),
        ("Triangulaire",         ratio_triangulaire, {"mode": tri_mode, "borne_max": tri_max}),
        ("Kumaraswamy(1.2,12)",  ratio_kumaraswamy, {}),
        ("PERT",                 ratio_pert, {"mode": max(0.0, pert_mode), "maxi": pert_max}),
    ]

    print(f"   {'Loi':22s} | {'VaR 99,5%':>10s}")
    print("   " + "-" * 35)

    ref_var = None
    rng = np.random.default_rng(GRAINE)
    for nom, loi, kw in lois:
        L = simuler_sanction_loi(loi, plafond, M, rng, **kw)
        var = np.quantile(L, NIVEAU_VAR)
        if ref_var is None: ref_var = var
        ecart = 100 * (var - ref_var) / ref_var if ref_var else 0.0
        marque = "ref" if "REF" in nom else f"{ecart:+.0f}%"
        print(f"   {nom:22s} | {var/1e6:7.2f} M€  ({marque})")

if __name__ == "__main__":
    main()