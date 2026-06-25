# -*- coding: utf-8 -*-
"""
=============================================================================
BRIQUE AGGRAVATION — contrefactuel (coûts indirects du non-respect)
=============================================================================
Surcoût qui n'aurait pas été subi avec un dispositif DORA conforme.
Modèle : Bernoulli(p) * Lognormale(mu, sigma). Dire d'expert.
"""

import numpy as np


def calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2):
    """Calibre la brique aggravation (jugement expert)."""
    return {
        "source": "Jugement expert (coûts indirects du non-respect)",
        "statut": "Hypothèses",
        "proba_survenance": float(proba_survenance),
        "mu": float(mu),
        "sigma": float(sigma),
    }


def simuler_aggravation(cal, M, rng):
    """Simule M années de pertes d'aggravation."""
    survient = rng.random(size=M) <= cal["proba_survenance"]
    severites = rng.lognormal(mean=cal["mu"], sigma=cal["sigma"], size=M)
    return np.where(survient, severites, 0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    cal = calibrer_aggravation()
    L = simuler_aggravation(cal, 500_000, rng)
    print("BRIQUE AGGRAVATION")
    print(f"  proba={cal['proba_survenance']:.0%} lognorm(mu={cal['mu']}, sigma={cal['sigma']})")
    print(f"  VaR 99,5% = {np.quantile(L, 0.995)/1e6:.2f} M€")
