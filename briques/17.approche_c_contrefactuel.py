# -*- coding: utf-8 -*-
"""
=============================================================================
APPROCHE C — CONTREFACTUEL DORA (Conforme vs Non-conforme)
=============================================================================

Mesure la perte imputable au NON-RESPECT de DORA comme l'écart de capital
entre deux états du monde simulés sur un PROFIL DE RISQUE IDENTIQUE :

    Delta_DORA = VaR_99,5%(L | non conforme) - VaR_99,5%(L | conforme)

La conformité n'est pas une perte directe, elle agit sur les PARAMETRES :
  - Fréquence (lambda) : baisse via l'hygiène cyber.
  - Dépendance (theta) : baisse via la redondance et les stratégies de sortie.

STATUT : Borne supérieure basée sur un jugement d'expert quant à l'efficacité 
des mesures DORA. Modélisation vectorisée ultra-rapide.
"""

import numpy as np
import brique_remediation_corrigee as b_rem
import brique_sanction as b_sanc
import brique_prestataire as b_pres
import brique_aggravation as b_agg
import copule_gumbel_agregation as copule
from ancrage_bilan_plausibilite import profil_assureur

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

# --- HYPOTHESES D'EFFICACITE DE LA CONFORMITE (dire d'expert) ---
# Fréquence (Brique Remédiation)
LAMBDA_NON_CONF = 3.0          
LAMBDA_CONF = 1.5              # -50% d'incidents réussis

# Dépendance de queue (Copule)
THETA_NON_CONF = 1.8           # Forte contagion (ex: panne prestataire entraîne le reste)
THETA_CONF = 1.2               # Faible contagion (résilience, PRA/PCA, sortie)

def simuler_monde(lam, theta, rng):
    """Simule L_DORA pour un état du monde avec la MÊME graine."""
    # Marginales fixes
    cal_s = b_sanc.calibrer_sanction(ca_annuel=800e6)
    cal_p = b_pres.calibrer_prestataire()
    cal_a = b_agg.calibrer_aggravation()
    
    L_s = b_sanc.simuler_sanction(cal_s, M, rng)
    L_p = b_pres.simuler_prestataire(cal_p, M, rng)
    L_a = b_agg.simuler_aggravation(cal_a, M, rng)
    
    # Marginale variable (Fréquence)
    cal_r = b_rem.calibrer_remediation(lambda_freq=lam, plafond_individuel=40_000_000.0)
    L_r = b_rem.simuler_remediation(cal_r, M, rng)
    
    marginales = [L_r, L_p, L_s, L_a]
    
    # Agrégation (Dépendance variable)
    U = copule.echantillon_gumbel(M, 4, theta, rng)
    L_dora = np.zeros(M)
    for j in range(4):
        L_dora += copule.make_quantile_empirique(marginales[j])(U[:, j])
        
    return np.quantile(L_dora, NIVEAU_VAR)

def main():
    profil = profil_assureur(ca_annuel=800e6)

    print("=" * 74)
    print(" APPROCHE C — CONTREFACTUEL DORA (Profil Constant)")
    print("=" * 74)
    print(f"   Hypothèses d'efficacité de la conformité DORA :")
    print(f"     Fréquence  : lambda {LAMBDA_NON_CONF:.1f} -> {LAMBDA_CONF:.1f} (-50%)")
    print(f"     Dépendance : theta  {THETA_NON_CONF:.1f} -> {THETA_CONF:.1f} (Baisse contagion)")

    # MEME graine pour neutraliser le biais de sélection
    rng_nc = np.random.default_rng(GRAINE)
    var_nc = simuler_monde(LAMBDA_NON_CONF, THETA_NON_CONF, rng_nc)
    
    rng_c = np.random.default_rng(GRAINE)
    var_c = simuler_monde(LAMBDA_CONF, THETA_CONF, rng_c)

    delta = var_nc - var_c

    # Décomposition de l'effet
    rng_freq = np.random.default_rng(GRAINE)
    var_freq_only = simuler_monde(LAMBDA_CONF, THETA_NON_CONF, rng_freq)
    effet_freq = var_nc - var_freq_only
    effet_theta = delta - effet_freq

    print("\n" + "=" * 74)
    print(" RESULTATS DU GAIN EN CAPITAL")
    print("=" * 74)
    print(f"   SCR_DORA (Non Conforme) : {var_nc/1e6:6.1f} M€")
    print(f"   SCR_DORA (Conforme)     : {var_c/1e6:6.1f} M€")
    print("   " + "-" * 52)
    print(f"   => Delta_DORA (GAIN)    : {delta/1e6:6.1f} M€ d'économie de capital")
    print(f"      Soit une réduction   : {100*delta/var_nc:6.1f} %")

    print("\n   Decomposition de l'économie :")
    print(f"     - Gain lié à l'hygiène cyber (fréquence) : {effet_freq/1e6:6.1f} M€ ({100*effet_freq/delta:3.0f}%)")
    print(f"     - Gain lié aux plans de sortie (theta)   : {effet_theta/1e6:6.1f} M€ ({100*effet_theta/delta:3.0f}%)")

    print("\n" + "=" * 74)
    print(" ARGUMENTAIRE COMMERCIAL POUR NEXIALOG")
    print("=" * 74)
    print(f"""   - La mise en conformité DORA n'est pas qu'un centre de coût.
   - Elle permet de libérer ~{delta/1e6:.0f} M€ de capital réglementaire (SCR).
   - Ce montant constitue le "Retour sur Investissement" prudentiel des
     projets de résilience cyber menés chez les clients.""")
    print("=" * 74)

if __name__ == "__main__":
    main()