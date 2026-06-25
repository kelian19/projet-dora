# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITÉ STRUCTURELLE DU SCR GLOBAL (Choix de la dépendance)
=============================================================================
"""

import numpy as np
import brique_remediation_corrigee as b_rem
import brique_sanction as b_sanc
import brique_prestataire as b_pres
import brique_aggravation as b_agg
import copule_gumbel_agregation as copule

NIVEAU_VAR = 0.995
M = 500_000

def main():
    rng = np.random.default_rng(42)
    
    # 1. Génération des marginales standards
    L_s = b_sanc.simuler_sanction(b_sanc.calibrer_sanction(), M, rng)
    L_p = b_pres.simuler_prestataire(b_pres.calibrer_prestataire(), M, rng)
    L_a = b_agg.simuler_aggravation(b_agg.calibrer_aggravation(), M, rng)
    L_r = b_rem.simuler_remediation(b_rem.calibrer_remediation(), M, rng)
    
    marginales = [L_s, L_r, L_p, L_a]
    
    print("=" * 70)
    print(" SENSIBILITÉ DE L'AGRÉGATION (Dépendance de Queue)")
    print("=" * 70)

    # Indépendance
    scr_indep = np.quantile(np.sum(marginales, axis=0), NIVEAU_VAR)
    print(f"   1. Indépendance Stricte (Theta = 1.0) : {scr_indep/1e6:6.1f} M€")
    
    # Gumbel avec différentes forces
    for theta in [1.2, 1.5, 2.0]:
        U = copule.echantillon_gumbel(M, 4, theta, rng)
        L_gumbel = np.zeros(M)
        for j in range(4):
            L_gumbel += copule.make_quantile_empirique(marginales[j])(U[:, j])
        scr_gumbel = np.quantile(L_gumbel, NIVEAU_VAR)
        tag = " <- Central" if theta == 1.5 else ""
        print(f"   2. Copule Gumbel      (Theta = {theta:<3.1f}) : {scr_gumbel/1e6:6.1f} M€{tag}")
        
    # Comonotonie
    scr_como = sum([np.quantile(m, NIVEAU_VAR) for m in marginales])
    print(f"   3. Comonotonie absolue (Pire absolu)  : {scr_como/1e6:6.1f} M€")

if __name__ == "__main__":
    main()