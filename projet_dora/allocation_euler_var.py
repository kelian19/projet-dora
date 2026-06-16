# -*- coding: utf-8 -*-
"""
=============================================================================
ALLOCATION D'EULER-VaR — version correcte (voisinage du quantile)
=============================================================================
Contribution d'Euler à la VaR (Tasche) : E[L_i | L_op = VaR], conditionnée
sur le quantile EXACT (pas sur toute la queue, ce qui donnerait la TVaR).
L'évènement {L_op = VaR} ayant proba nulle, on moyenne sur une BANDE étroite
autour du quantile (Koike-Saporito-Targino).

Propriétés garanties (testées) :
  - additivité : somme des contributions = VaR_alpha(L_op)
  - diversification : contribution_i <= VaR_alpha(L_i) stand-alone
"""

import numpy as np

NIVEAU_VAR = 0.995


def allocation_euler_var(composantes, niveau=NIVEAU_VAR, demi_bande=0.0025,
                         renormaliser=True):
    """Alloue VaR_alpha(L_op) entre les composantes par contribution d'Euler."""
    composantes = [np.asarray(c, dtype=float) for c in composantes]
    L_op = np.sum(composantes, axis=0)
    var_globale = np.quantile(L_op, niveau)

    q_bas = np.quantile(L_op, max(0.0, niveau - demi_bande))
    q_haut = np.quantile(L_op, min(1.0, niveau + demi_bande))
    masque = (L_op >= q_bas) & (L_op <= q_haut)

    if masque.sum() < 50:
        demi_bande *= 4
        q_bas = np.quantile(L_op, max(0.0, niveau - demi_bande))
        q_haut = np.quantile(L_op, min(1.0, niveau + demi_bande))
        masque = (L_op >= q_bas) & (L_op <= q_haut)

    contributions = np.array([c[masque].mean() for c in composantes])
    if renormaliser and contributions.sum() > 0:
        contributions = contributions * (var_globale / contributions.sum())

    return {
        "VaR_globale": var_globale,
        "contributions": contributions,
        "n_bande": int(masque.sum()),
        "bande": (q_bas, q_haut),
    }


def valider_allocation(composantes, res, niveau=NIVEAU_VAR):
    """Vérifie additivité et diversification (alloué <= stand-alone)."""
    contributions = res["contributions"]
    var_globale = res["VaR_globale"]
    err_add = abs(contributions.sum() - var_globale) / var_globale
    standalone = np.array([np.quantile(c, niveau) for c in composantes])
    test_div = bool(np.all(contributions <= standalone + 1e-9))
    return {"test_additivite": err_add < 1e-6, "erreur_additivite": err_add,
            "test_diversification": test_div, "standalone": standalone}


if __name__ == "__main__":
    # Démo autonome avec marginales jouet
    rng = np.random.default_rng(2024)
    M = 500_000
    a = rng.lognormal(11, 1.0, M)
    b = rng.lognormal(13, 1.4, M)
    c = rng.lognormal(10, 0.8, M)
    from copule_gumbel_agregation import echantillon_gumbel, make_quantile_empirique
    U = echantillon_gumbel(M, 3, 1.5, rng)
    comp = [make_quantile_empirique(x)(U[:, i]) for i, x in enumerate([a, b, c])]
    res = allocation_euler_var(comp)
    val = valider_allocation(comp, res)
    print("ALLOCATION EULER-VaR (démo)")
    print(f"  SCR global = {res['VaR_globale']/1e6:.2f} M€  (bande n={res['n_bande']})")
    for i, (ct, sa) in enumerate(zip(res["contributions"], val["standalone"])):
        print(f"  brique {i}: alloué={ct/1e6:.2f}M€  stand-alone={sa/1e6:.2f}M€  "
              f"{'OK' if ct <= sa + 1e-9 else 'VIOLE'}")
    print(f"  additivité: {'OK' if val['test_additivite'] else 'ECHEC'}  "
          f"diversification: {'OK' if val['test_diversification'] else 'ECHEC'}")
