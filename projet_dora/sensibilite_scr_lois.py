# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITE AUX LOIS DE FREQUENCE — POISSON vs NEGBIN
=============================================================================
Démontre pourquoi la Binomiale Négative (qui autorise une surdispersion, 
c'est-à-dire Variance > Moyenne) est la nouvelle référence du modèle au lieu 
de la Poisson (Variance = Moyenne).

La NegBin capture le "clustering" des cyberattaques (une faille zero-day 
entraîne souvent une rafale d'incidents la même année).
"""
import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

LAMBDA_AN = 2.0
FACTEUR_SURDISPERSION = 2.0  # Variance = 2 * Moyenne
PLAFOND = 40_000_000.0   # plafond central (40 M€, ~5% FP, aligné sur l'approche A)

# Paramètres Sévérité
MEDIANE = 171_000.0
Q95 = 9_340_000.0
U_POT = 4_950_000.0
XI = 1.3
BETA = 2_000_000.0
P_QUEUE = 0.10

def simuler_remediation(rng, freq_mode="negbin"):
    mu = np.log(MEDIANE)
    sigma = (np.log(Q95) - mu) / stats.norm.ppf(0.95)
    
    if freq_mode == "negbin":
        # p = mean / variance = 1 / facteur = 0.5
        # n = mean * p / (1-p) = lambda
        n_incidents = rng.negative_binomial(LAMBDA_AN, 1/FACTEUR_SURDISPERSION, size=M)
    else: # Poisson
        n_incidents = rng.poisson(LAMBDA_AN, size=M)
        
    tot = int(n_incidents.sum())
    en_q = rng.random(tot) < P_QUEUE
    sev = np.empty(tot)
    nq = int(en_q.sum())
    
    if nq > 0:
        sev[en_q] = U_POT + stats.genpareto.rvs(c=XI, scale=BETA, size=nq, random_state=rng)
    if tot - nq > 0:
        sev[~en_q] = rng.lognormal(mu, sigma, size=tot - nq)
        
    sev = np.minimum(sev, PLAFOND)
    idx = np.cumsum(n_incidents)[:-1]
    return np.array([b.sum() if len(b) > 0 else 0.0 for b in np.split(sev, idx)])

def main():
    print("=" * 74)
    print(" SENSIBILITE FREQUENCE : POISSON vs BINOMIALE NEGATIVE [REF]")
    print("=" * 74)
    
    rng = np.random.default_rng(GRAINE)
    L_negbin = simuler_remediation(rng, freq_mode="negbin")
    
    rng = np.random.default_rng(GRAINE)
    L_poisson = simuler_remediation(rng, freq_mode="poisson")
    
    var_nb = np.quantile(L_negbin, NIVEAU_VAR)
    var_p = np.quantile(L_poisson, NIVEAU_VAR)
    
    print(f"   {'Distribution de Fréquence':35s} | {'VaR 99.5% (Remédiation)':>25s}")
    print("   " + "-" * 63)
    print(f"   {'Binomiale Négative (Var=2xMoy) [REF]':35s} | {var_nb/1e6:22.2f} M€")
    print(f"   {'Poisson (Var=Moy)':35s} | {var_p/1e6:22.2f} M€")
    print("   " + "-" * 63)
    print(f"""\n   Analyse : La NegBin intègre le regroupement des sinistres. 
   À moyenne égale, le risque de subir "une très mauvaise année" avec 
   de multiples attaques est plus élevé, gonflant prudemment le capital.""")

if __name__ == "__main__":
    main()