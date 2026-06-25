# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITÉ : BRIQUE SANCTION
=============================================================================
Test de la sensibilité face à la probabilité qu'une faille TIC entraîne
une véritable enquête et sanction de l'ACPR.
"""

import numpy as np
import brique_sanction as b_sanc

def main():
    M = 500_000
    rng = np.random.default_rng(42)
    
    print("=" * 70)
    print(" SENSIBILITÉ DE LA BRIQUE SANCTION (Plafond fixe à 16 M€)")
    print("=" * 70)
    
    for proba in [0.05, 0.10, 0.20]:
        cal = b_sanc.calibrer_sanction(proba_survenance=proba)
        L = b_sanc.simuler_sanction(cal, M, rng)
        print(f"   Probabilité d'amende ACPR = {proba*100:>2.0f}% -> VaR 99.5% = {np.quantile(L, 0.995)/1e6:4.1f} M€")

if __name__ == "__main__":
    main()