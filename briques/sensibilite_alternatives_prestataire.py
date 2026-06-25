# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITÉ : BRIQUE PRESTATAIRE (Jugement Expert)
=============================================================================
Évalue l'impact d'une erreur d'estimation des experts sur les quantiles 
des scénarios catastrophes (ex: pannes cloud sous-estimées).
"""

import numpy as np
import brique_prestataire as b_pres

def main():
    M = 500_000
    rng = np.random.default_rng(42)
    
    scenarios_base = b_pres.calibrer_prestataire()
    
    print("=" * 70)
    print(" SENSIBILITÉ DE LA BRIQUE PRESTATAIRE AUX QUANTILES EXPERTS")
    print("=" * 70)
    
    for multiplicateur in [0.70, 1.00, 1.30]:
        scenarios_modifies = []
        # On recrée les scénarios de base avec le multiplicateur de stress
        scenarios_bruts = [
            {"nom": "Panne cloud", "proba_an": 0.08, "q50": 2e6, "q95": 25e6, "q995": 150e6, "loi": "gpd"},
            {"nom": "Compromission paiement", "proba_an": 0.05, "q50": 1e6, "q95": 12e6, "q995": 80e6, "loi": "gpd"},
            {"nom": "Ransomware éditeur", "proba_an": 0.12, "q50": 3e5, "q95": 3e6, "q995": 20e6, "loi": "lognorm"}
        ]
        for s in scenarios_bruts:
            s_mod = dict(s)
            s_mod["q95"] *= multiplicateur
            s_mod["q995"] *= multiplicateur
            scenarios_modifies.append(s_mod)
            
        cal = b_pres.calibrer_prestataire(scenarios_modifies)
        L = b_pres.simuler_prestataire(cal, M, rng)
        
        signe = "+" if multiplicateur > 1 else ("-" if multiplicateur < 1 else " ")
        print(f"   Quantiles experts {signe}{abs(1-multiplicateur)*100:>2.0f}% -> VaR 99.5% = {np.quantile(L, 0.995)/1e6:5.1f} M€")

if __name__ == "__main__":
    main()