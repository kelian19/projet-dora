# -*- coding: utf-8 -*-
"""
=============================================================================
BRIQUE REMEDIATION — VERSION CORRIGEE (corps + queue POT)
=============================================================================

CORRECTION D'UN DEFAUT MAJEUR de la version précédente
------------------------------------------------------
L'ancienne brique faisait `severite = seuil_u + GPD(...)` pour TOUS les
sinistres, avec seuil_u = 1,86 M€. Conséquence : chaque incident TIC coûtait
au minimum 1,86 M€ (plancher absurde), la sévérité moyenne atteignait 12,6 M€
et la VaR explosait à ~430 M€ pour la seule remédiation -> SCR_DORA ~2000 M€,
soit 2,5x le CA de l'entité. Non plausible.

CAUSE : confusion entre le SEUIL POT (u, au-dessus duquel on modélise la
queue) et un PLANCHER de coût. En approche POT, seule la QUEUE (les ~5% de
sinistres extrêmes) suit `u + GPD`. Les 95% de sinistres du CORPS sont petits
et suivent une loi distincte (lognormale). L'ancienne version avait oublié le
corps et appliquait le mécanisme de queue partout.

STRUCTURE CORRECTE (mélange corps/queue)
----------------------------------------
  - avec proba (1 - p_queue) : sinistre du CORPS ~ Lognormale(mu, sigma)
  - avec proba p_queue        : sinistre de QUEUE ~ u + GPD(xi, beta)
  - plafond individuel ANCRE sur le bilan (perte max réaliste d'une entité)

Le paramètre xi (forme GPD) reste celui de la littérature cyber [1,1 ; 1,5] ;
sa valeur est testée par analyse de sensibilité, pas estimée (queue trop creuse).

AVERTISSEMENT : paramètres du corps et seuil = hypothèses ancrées sur des
ordres de grandeur d'incidents cyber, pas une estimation sur données.
"""

import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995


# =============================================================================
# CALIBRATION
# =============================================================================
def calibrer_remediation(
    n_incidents_dora=17,
    annees_historiques=10.0,
    # --- corps de distribution (petits/moyens sinistres) ---
    mediane_corps=120_000.0,
    q95_corps=2_000_000.0,
    # --- queue POT (sinistres extrêmes) ---
    xi=1.3,
    p_queue=0.05,
    beta=800_000.0,
    # --- plafond individuel ancré sur le bilan ---
    plafond_individuel=196_000_000.0,
):
    """
    Calibre la brique remédiation avec structure corps + queue.

    mediane_corps, q95_corps : ancrent la lognormale du corps (petits sinistres).
    xi, beta, p_queue        : queue GPD au-dessus du seuil u = q95_corps.
    plafond_individuel       : perte max réaliste par sinistre (ancrage bilan).
    """
    lambda_poisson = n_incidents_dora / annees_historiques

    # Lognormale du corps calibrée sur (médiane, q95)
    mu = np.log(mediane_corps)
    sigma = (np.log(q95_corps) - mu) / stats.norm.ppf(0.95)
    seuil_u = q95_corps  # le seuil POT = q95 du corps (continuité corps/queue)

    return {
        "source": "Corps lognormal (ordres de grandeur cyber) + queue GPD littérature",
        "statut": "Scénario central (xi repris littérature, testé en sensibilité)",
        "lambda_poisson": float(lambda_poisson),
        "mu_corps": float(mu),
        "sigma_corps": float(sigma),
        "seuil_u": float(seuil_u),
        "xi": float(xi),
        "beta": float(beta),
        "p_queue": float(p_queue),
        "plafond_individuel": float(plafond_individuel),
        "n_incidents": int(n_incidents_dora),
        "annees_historiques": float(annees_historiques),
    }


# =============================================================================
# SIMULATION
# =============================================================================
def simuler_remediation(cal, M, rng, diagnostic=False):
    """Simule M années de pertes de remédiation (structure corps + queue)."""
    n_par_an = rng.poisson(cal["lambda_poisson"], size=M)
    tot = int(n_par_an.sum())
    if tot == 0:
        return np.zeros(M)

    # Affectation corps / queue
    en_queue = rng.random(tot) < cal["p_queue"]
    sev = np.empty(tot)

    # Queue : u + GPD(xi, beta)
    nq = int(en_queue.sum())
    if nq > 0:
        sev[en_queue] = cal["seuil_u"] + stats.genpareto.rvs(
            c=cal["xi"], scale=cal["beta"], size=nq, random_state=rng)

    # Corps : lognormale(mu, sigma)
    nc = tot - nq
    if nc > 0:
        sev[~en_queue] = rng.lognormal(cal["mu_corps"], cal["sigma_corps"], size=nc)

    # Plafond individuel ancré
    sev = np.minimum(sev, cal["plafond_individuel"])

    # Agrégation annuelle
    idx = np.cumsum(n_par_an)[:-1]
    blocs = np.split(sev, idx)
    L = np.array([b.sum() if len(b) > 0 else 0.0 for b in blocs])

    if diagnostic:
        print(f"  sév. indiv : médiane={np.median(sev)/1e3:.0f}k€ "
              f"moy={sev.mean()/1e6:.2f}M€ q99={np.quantile(sev,.99)/1e6:.2f}M€ "
              f"max={sev.max()/1e6:.1f}M€")
        print(f"  L_rem      : moy={L.mean()/1e6:.2f}M€ "
              f"VaR99.5={np.quantile(L,NIVEAU_VAR)/1e6:.1f}M€")

    return L


# =============================================================================
# TEST LOCAL + SENSIBILITE XI
# =============================================================================
if __name__ == "__main__":
    M = 500_000
    rng = np.random.default_rng(42)

    print("=" * 72)
    print(" BRIQUE REMEDIATION CORRIGEE — scénario central + sensibilité xi")
    print("=" * 72)
    for xi in [1.1, 1.3, 1.5]:
        cal = calibrer_remediation(xi=xi)
        L = simuler_remediation(cal, M, rng)
        var = np.quantile(L, NIVEAU_VAR)
        print(f"  xi={xi:.1f} | VaR99.5 remédiation = {var/1e6:7.2f} M€ "
              f"| moyenne = {L.mean()/1e6:5.2f} M€")
    print("=" * 72)
    print("  Comparaison : ancienne structure (seuil_u plancher) -> ~430 M€")
    print("  Nouvelle structure (corps+queue)                    -> ~30-50 M€")
