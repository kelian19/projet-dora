# -*- coding: utf-8 -*-
"""
=============================================================================
BRIQUE PRESTATAIRE — analyse de scénarios structurés (dire d'expert)
=============================================================================
Pas d'historique de défaillance de prestataire critique -> jugement d'expert
structuré (admissible ORSA, hérité du cadre AMA bancaire). Chaque scénario
catastrophe est calibré sur 3 quantiles experts {q0.5, q0.95, q0.995}.
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq


def calibrer_lognormale_2q(q50, q95):
    """Calibre une lognormale sur médiane et q95."""
    z50, z95 = stats.norm.ppf(0.50), stats.norm.ppf(0.95)
    mu = np.log(q50)
    sigma = (np.log(q95) - mu) / (z95 - z50)
    return mu, sigma


def calibrer_gpd_3q(q50, q95, q995, seuil=None):
    """Calibre une GPD passant par (q95, q995) au-dessus du seuil u = q50."""
    u = q50 if seuil is None else seuil
    y1, y2 = q95 - u, q995 - u
    p1, p2 = 0.90, 0.99

    def equation(xi):
        if abs(xi) < 1e-8:
            return (y2 / y1) - (np.log(1 - p2) / np.log(1 - p1))
        lhs = ((1 - p2) ** (-xi) - 1) / ((1 - p1) ** (-xi) - 1)
        return lhs - (y2 / y1)

    try:
        xi = brentq(equation, 0.01, 3.0)
    except ValueError:
        xi = 1.0
    beta = y1 * xi / ((1 - p1) ** (-xi) - 1)
    return float(xi), float(beta), float(u)


def calibrer_prestataire(scenarios=None):
    """Calibre la brique prestataire à partir de scénarios experts."""
    if scenarios is None:
        scenarios = [
            {"nom": "Panne hyperscaler (cloud)", "proba_an": 0.08,
             "q50": 2_000_000, "q95": 25_000_000, "q995": 150_000_000, "loi": "gpd"},
            {"nom": "Compromission prestataire paiement", "proba_an": 0.05,
             "q50": 1_000_000, "q95": 12_000_000, "q995": 80_000_000, "loi": "gpd"},
            {"nom": "Rançongiciel éditeur métier", "proba_an": 0.12,
             "q50": 300_000, "q95": 3_000_000, "q995": 20_000_000, "loi": "lognorm"},
        ]

    calibres = []
    for s in scenarios:
        if s["loi"] == "lognorm":
            mu, sigma = calibrer_lognormale_2q(s["q50"], s["q95"])
            calibres.append({"nom": s["nom"], "proba_an": float(s["proba_an"]),
                             "loi": "lognorm", "mu": float(mu), "sigma": float(sigma),
                             "source": "Scénario expert", "statut": "Hypothèse"})
        else:
            xi, beta, u = calibrer_gpd_3q(s["q50"], s["q95"], s["q995"])
            calibres.append({"nom": s["nom"], "proba_an": float(s["proba_an"]),
                             "loi": "gpd", "xi": xi, "beta": beta, "u": u,
                             "source": "Scénario expert", "statut": "Hypothèse"})
    return calibres


def simuler_prestataire(scenarios_calibres, M, rng, plafond=2_000_000_000.0):
    """Simule M années : L_presta = somme des scénarios survenus."""
    L = np.zeros(M)
    for s in scenarios_calibres:
        survient = rng.random(M) < s["proba_an"]
        n = int(survient.sum())
        if n == 0:
            continue
        if s["loi"] == "lognorm":
            sev = rng.lognormal(s["mu"], s["sigma"], size=n)
        else:
            v = rng.random(n)
            sev = s["u"] + (s["beta"] / s["xi"]) * ((1 - v) ** (-s["xi"]) - 1)
        L[survient] += np.minimum(sev, plafond)
    return L


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    cal = calibrer_prestataire()
    L = simuler_prestataire(cal, 500_000, rng)
    print("BRIQUE PRESTATAIRE")
    for s in cal:
        if s["loi"] == "gpd":
            print(f"  {s['nom']:38s} GPD xi={s['xi']:.2f} p={s['proba_an']:.0%}")
        else:
            print(f"  {s['nom']:38s} LN  mu={s['mu']:.1f} p={s['proba_an']:.0%}")
    print(f"  VaR 99,5% = {np.quantile(L, 0.995)/1e6:.2f} M€")
