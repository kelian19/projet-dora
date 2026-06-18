# -*- coding: utf-8 -*-
"""
=============================================================================
BRIQUE REMEDIATION — VERSION CORRIGEE ET ALIGNEE (corps + queue POT)
=============================================================================

STRUCTURE CORRECTE (mélange corps/queue)
----------------------------------------
  - avec proba (1 - p_queue) : sinistre du CORPS ~ Lognormale(mu, sigma)
  - avec proba p_queue       : sinistre de QUEUE ~ u + GPD(xi, beta)
  - plafond individuel ANCRE sur le bilan (perte max réaliste d'une entité)

ALIGNEMENT :
Cette version intègre la Binomiale Négative (surdispersion) pour modéliser
le regroupement des attaques (clustering), et utilise par défaut les 
paramètres calibrés sur la base de données réelle PRC (incidents HACK).
"""

import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995


# =============================================================================
# CALIBRATION
# =============================================================================
def calibrer_remediation(
    # --- Fréquence (Binomiale Négative) ---
    lambda_freq=2.0,
    facteur_surdispersion=2.0,
    # --- Corps de distribution (PRC Hack) ---
    mediane_corps=171_000.0,
    q95_corps=9_340_000.0,
    # --- Queue POT (sinistres extrêmes) ---
    seuil_pot=4_950_000.0,
    xi=1.3,
    p_queue=0.10,
    beta=2_000_000.0,
    # --- Plafond individuel ancré sur le bilan ---
    plafond_individuel=50_000_000.0,
):
    """
    Calibre la brique remédiation avec structure corps + queue et fréquence NegBin.
    Les paramètres par défaut correspondent au scénario central du modèle global.
    """
    # Lognormale du corps calibrée sur (médiane, q95)
    mu = np.log(mediane_corps)
    sigma = (np.log(q95_corps) - mu) / stats.norm.ppf(0.95)

    return {
        "source": "Corps lognormal (PRC) + queue GPD + Fréquence NegBin",
        "statut": "Scénario central",
        "lambda_freq": float(lambda_freq),
        "facteur_surdispersion": float(facteur_surdispersion),
        "mu_corps": float(mu),
        "sigma_corps": float(sigma),
        "seuil_u": float(seuil_pot),
        "xi": float(xi),
        "beta": float(beta),
        "p_queue": float(p_queue),
        "plafond_individuel": float(plafond_individuel),
    }


# =============================================================================
# SIMULATION
# =============================================================================
def simuler_remediation(cal, M, rng, diagnostic=False):
    """Simule M années de pertes de remédiation (fréquence NegBin + sévérité Spliced)."""
    
    # 1. Tirage de la fréquence (Binomiale Négative)
    # p = moyenne / variance = 1 / facteur_surdispersion
    p_negbin = 1.0 / cal["facteur_surdispersion"]
    n_par_an = rng.negative_binomial(cal["lambda_freq"], p_negbin, size=M)
    
    tot = int(n_par_an.sum())
    if tot == 0:
        return np.zeros(M)

    # 2. Affectation corps / queue
    en_queue = rng.random(tot) < cal["p_queue"]
    sev = np.empty(tot)

    # 3. Sévérité de Queue : u + GPD(xi, beta)
    nq = int(en_queue.sum())
    if nq > 0:
        sev[en_queue] = cal["seuil_u"] + stats.genpareto.rvs(
            c=cal["xi"], scale=cal["beta"], size=nq, random_state=rng)

    # 4. Sévérité du Corps : lognormale(mu, sigma)
    nc = tot - nq
    if nc > 0:
        sev[~en_queue] = rng.lognormal(cal["mu_corps"], cal["sigma_corps"], size=nc)

    # 5. Plafond individuel ancré
    sev = np.minimum(sev, cal["plafond_individuel"])

    # 6. Agrégation annuelle
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
    print("  Ce test isolé confirme que la brique tourne correctement avec la")
    print("  nouvelle fréquence Binomiale Négative et les paramètres PRC.")