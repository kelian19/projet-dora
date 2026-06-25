# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITÉ DE Delta_DORA AUX HYPOTHÈSES D'EFFICACITÉ DE LA CONFORMITÉ
=============================================================================
Plutôt que de présenter Delta_DORA comme un point unique, ce module
construit une GRILLE pour plusieurs jeux d'hypothèses d'efficacité.
"""

import numpy as np
import brique_remediation_corrigee as b_rem
import brique_sanction as b_sanc
import brique_prestataire as b_pres
import brique_aggravation as b_agg
import copule_gumbel_agregation as copule

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

LAMBDA_NON_CONF = 3.0
THETA_NON_CONF = 1.8

def simuler_VaR(lam, theta, graine=GRAINE):
    """VaR 99,5% du SCR_DORA agrégé pour un jeu (lambda, theta) donné."""
    rng = np.random.default_rng(graine)
    
    cal_s = b_sanc.calibrer_sanction(ca_annuel=800e6)
    cal_p = b_pres.calibrer_prestataire()
    cal_a = b_agg.calibrer_aggravation()
    cal_r = b_rem.calibrer_remediation(lambda_freq=lam, plafond_individuel=40_000_000.0)

    L_s = b_sanc.simuler_sanction(cal_s, M, rng)
    L_p = b_pres.simuler_prestataire(cal_p, M, rng)
    L_a = b_agg.simuler_aggravation(cal_a, M, rng)
    L_r = b_rem.simuler_remediation(cal_r, M, rng)
    
    U = copule.echantillon_gumbel(M, 4, theta, rng)
    marg = [L_s, L_r, L_p, L_a]
    
    L_dora = np.zeros(M)
    for j in range(4):
        L_dora += copule.make_quantile_empirique(marg[j])(U[:, j])
    return np.quantile(L_dora, NIVEAU_VAR)

def main():
    var_nc = simuler_VaR(LAMBDA_NON_CONF, THETA_NON_CONF)

    print("=" * 76)
    print(" SENSIBILITÉ DE Delta_DORA AUX HYPOTHÈSES D'EFFICACITÉ (Grille)")
    print("=" * 76)
    print(f"   État NON CONFORME (fixe) : lambda={LAMBDA_NON_CONF}, theta={THETA_NON_CONF} -> VaR = {var_nc/1e6:.1f} M€\n")

    reductions_freq = [0.30, 0.50, 0.70]      
    thetas_conf = [1.4, 1.2, 1.1]             

    entete = "   Réd. Fréq \\ Theta Cible |" + "".join(f"  Theta={t:<5.1f}" for t in thetas_conf)
    print(entete)
    print("   " + "-" * 64)

    grille = {}
    for rf in reductions_freq:
        lam_c = LAMBDA_NON_CONF * (1 - rf)
        ligne = f"   -{int(rf*100):>2d}% (lam={lam_c:.1f})      |"
        for tc in thetas_conf:
            var_c = simuler_VaR(lam_c, tc)
            delta = (var_nc - var_c) / 1e6
            grille[(rf, tc)] = delta
            ligne += f"  {delta:8.1f}"
        print(ligne)
    print("   " + "-" * 64)

    valeurs = np.array(list(grille.values()))
    centre = grille[(0.50, 1.2)]
    print(f"\n   Scénario central (-50% fréq, Theta=1.2) : {centre:.1f} M€")
    print(f"   Fourchette complète de la grille        : {valeurs.min():.1f} à {valeurs.max():.1f} M€")

if __name__ == "__main__":
    main()