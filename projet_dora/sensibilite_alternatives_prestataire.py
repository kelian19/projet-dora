# -*- coding: utf-8 -*-
"""
=============================================================================
BANC D'ESSAI DES ALTERNATIVES — BRIQUE PRESTATAIRE
=============================================================================
Confronte le modèle de référence (Jugement d'expert par 3 quantiles 
-> Lognormale) à la méthode standard de l'industrie : la loi PERT (Min, Mode, Max).

OBJECTIF : Démontrer pourquoi borner un risque cyber à un "Maximum expert" 
(méthode PERT) sous-estime le risque extrême, et pourquoi l'approche par 
quantiles (qui laisse la queue s'étendre) est plus prudente et réaliste.
"""

import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

# --- HYPOTHÈSES DE L'EXPERT (Scénario : Rançongiciel éditeur métier) ---
PROBA_AN = 0.12

# 1. Élicitation classique (PERT) : L'expert donne Min, Moyen, Max
PERT_MIN = 50_000.0
PERT_MODE = 300_000.0       # Le plus probable
PERT_MAX = 20_000_000.0     # "Le pire cas imaginable" selon l'expert

# 2. Élicitation probabiliste (Votre méthode) : L'expert donne des quantiles
# On fait correspondre le mode de la PERT à la médiane (Q50)
# On fait correspondre le pire cas (Max) au quantile à 99.5%
LOGNORM_Q50 = 300_000.0     
LOGNORM_Q995 = 20_000_000.0 

def simuler_pert(rng, n, mini, mode, maxi, lamb=4.0):
    """Génère un échantillon selon la loi PERT (Bêta reparamétrée)"""
    if maxi <= mini:
        return np.full(n, mini)
    alpha = 1.0 + lamb * (mode - mini) / (maxi - mini)
    beta_param = 1.0 + lamb * (maxi - mode) / (maxi - mini)
    x = rng.beta(alpha, beta_param, size=n)
    return mini + x * (maxi - mini)

def calibrer_lognormale(q50, q995):
    """Calibre mu et sigma à partir de la médiane et du quantile 99.5%"""
    z50, z995 = stats.norm.ppf(0.50), stats.norm.ppf(0.995)
    mu = np.log(q50)
    sigma = (np.log(q995) - mu) / (z995 - z50)
    return mu, sigma

def main():
    rng = np.random.default_rng(GRAINE)
    
    print("=" * 74)
    print(" SENSIBILITE AU CHOIX DE LOI — BRIQUE PRESTATAIRE")
    print("=" * 74)
    print(f"   Scénario : Rançongiciel sur un éditeur métier (Proba: {PROBA_AN:.0%}/an)")
    print(f"   Pire cas estimé par l'expert : {PERT_MAX/1e6:.1f} M€\n")
    
    # --- SIMULATION MÉTHODE PERT ---
    survient_pert = rng.random(M) < PROBA_AN
    sev_pert = simuler_pert(rng, M, PERT_MIN, PERT_MODE, PERT_MAX)
    L_pert = np.where(survient_pert, sev_pert, 0.0)
    
    # --- SIMULATION MÉTHODE LOGNORMALE (Notre approche) ---
    mu, sigma = calibrer_lognormale(LOGNORM_Q50, LOGNORM_Q995)
    survient_ln = rng.random(M) < PROBA_AN
    sev_ln = rng.lognormal(mu, sigma, size=M)
    L_ln = np.where(survient_ln, sev_ln, 0.0)

    # Note : Le VaR 99.5% agrégé avec proba = 0.12 correspond au quantile 
    # ~95.8% de la sévérité conditionnelle.
    var_pert = np.quantile(L_pert, NIVEAU_VAR)
    var_ln = np.quantile(L_ln, NIVEAU_VAR)
    
    max_pert = L_pert.max()
    max_ln = L_ln.max()
    
    col_methode = "Méthode d'élicitation"
    print(f"   {col_methode:35s} | {'VaR 99.5%':>12s} | {'Max simulé':>12s}")
    print("   " + "-" * 64)
    print(f"   {'PERT (Standard Industrie)':35s} | {var_pert/1e6:9.2f} M€ | {max_pert/1e6:9.2f} M€")
    print(f"   {'Lognormale par quantiles [REF]':35s} | {var_ln/1e6:9.2f} M€ | {max_ln/1e6:9.2f} M€")
    
    print("\n" + "=" * 74)
    print(" LECTURE POUR LA SOUTENANCE (L'argument pro-Quantiles)")
    print("=" * 74)
    print(f"""   - La méthode PERT borne strictement la perte : le modèle ne 
     générera JAMAIS un sinistre supérieur au "pire cas" de l'expert 
     ({PERT_MAX/1e6:.0f} M€).
   - Problème : En risque cyber ou cloud, l'expert sous-estime toujours 
     le "vrai" cygne noir (inconnu inconnu). Borner la distribution est un
     risque prudentiel majeur.
   - La méthode par quantiles (Lognormale/GPD) force l'expert à donner un 
     scénario à 1 sur 200 ans ({LOGNORM_Q995/1e6:.0f} M€), mais la courbe 
     continue d'évoluer au-delà vers l'infini (bridée uniquement par le 
     plafond de l'assureur). 
   => Conclusion : C'est la seule méthode qui permet au modèle interne 
      de générer des pertes dépassant l'imagination initiale des experts. 
      C'est indispensable pour capitaliser le risque systémique.""")
    print("=" * 74)

if __name__ == "__main__":
    main()