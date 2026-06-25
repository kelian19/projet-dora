# -*- coding: utf-8 -*-
"""
=============================================================================
BRIQUE SANCTION — proxy réglementaire
=============================================================================
Modèle : L_sanction = c * plafond, c ~ Beta(alpha, beta)
         plafond = 2% du CA annuel (régime entité financière, art. 50-52 DORA)

NB : ne PAS confondre avec l'astreinte de 1% du CA quotidien (art. 35) qui
vise les prestataires tiers critiques, ni avec le RGPD (4%). Le régime modélisé
ici est celui de l'entité financière elle-même.
"""

import numpy as np


def calibrer_sanction(
    df_registre=None,
    ca_annuel=800_000_000.0,
    taux_plafond=0.02,
    alpha_defaut=0.8,
    beta_defaut=6.0,
    proba_survenance=0.10,
):
    """
    Calibre la brique sanction.
    Si un registre contient des sanctions historiques (>=5), ajuste (alpha,beta)
    par méthode des moments sur le ratio amende/plafond ; sinon valeurs expertes.
    """
    plafond = taux_plafond * ca_annuel
    alpha, beta = alpha_defaut, beta_defaut
    source = "expert (Bêta par défaut)"

    if df_registre is not None and "sous_categorie" in df_registre.columns:
        sanctions = df_registre[
            df_registre["sous_categorie"].astype(str).str.contains("Sanction", na=False)
        ].copy()
        if "montant_eur" in sanctions.columns and len(sanctions) >= 5:
            ratios = sanctions["montant_eur"].values / plafond
            ratios = ratios[(ratios > 0) & (ratios < 1)]
            if len(ratios) >= 5:
                m, v = ratios.mean(), ratios.var(ddof=1)
                if 0 < v < m * (1 - m):
                    facteur = m * (1 - m) / v - 1
                    alpha = m * facteur
                    beta = (1 - m) * facteur
                    source = f"calibré sur {len(ratios)} sanctions du registre"

    return {
        "source": source,
        "statut": "Proxy réglementaire",
        "alpha": float(alpha),
        "beta": float(beta),
        "plafond": float(plafond),
        "proba_survenance": float(proba_survenance),
    }


def simuler_sanction(cal, M, rng):
    """Simule M réalisations annuelles de L_sanction."""
    survient = rng.random(M) < cal["proba_survenance"]
    ratios = rng.beta(cal["alpha"], cal["beta"], size=M)
    return np.where(survient, ratios * cal["plafond"], 0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    cal = calibrer_sanction()
    L = simuler_sanction(cal, 500_000, rng)
    print("BRIQUE SANCTION")
    print(f"  plafond (2% CA) = {cal['plafond']/1e6:.1f} M€")
    print(f"  Beta({cal['alpha']:.2f}, {cal['beta']:.2f})  proba={cal['proba_survenance']:.0%}")
    print(f"  VaR 99,5% = {np.quantile(L, 0.995)/1e6:.2f} M€")
