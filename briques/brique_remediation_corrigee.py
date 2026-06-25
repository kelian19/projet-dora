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

ALIGNEMENT PRC 2025 + JACOBS :
Cette version intègre les paramètres empiriques validés sur la base PRC 2025,
convertis en Euros via le modèle log-log de Jacobs (pente b=0.76).
L'indice de queue effectif sur les coûts ressort à xi = 0.988.
La fréquence utilise une Binomiale Négative pour modéliser le clustering.
"""

import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995

# =============================================================================
# CALIBRATION
# =============================================================================
def calibrer_remediation(
    # --- Fréquence (Binomiale Négative) ---
    # Fréquence cible pour l'entité évaluée
    lambda_freq=2.0,
    # Facteur de surdispersion conservateur (1.30)
    facteur_surdispersion=1.30,
    
    # --- Corps de distribution (Conversion PRC -> Euros via Jacobs) ---
    # Valeurs converties depuis les quantiles de volume
    mediane_corps=603_000.0,
    q95_corps=22_195_000.0,
    
    # --- Queue POT (Sinistres extrêmes) ---
    # Seuil u converti en euros (pour U_vol = 128 467)
    seuil_pot=15_310_000.0,
    # xi effectif sur les coûts (xi_volume * pente Jacobs = 1.30 * 0.76)
    xi=0.988,
    p_queue=0.0654,
    # beta ajusté pour respecter la forme de la queue convertie
    beta=21_000_000.0,
    
    # --- Plafond individuel ancré sur le bilan ---
    plafond_individuel=40_000_000.0,
):
    """
    Calibre la brique remédiation avec structure corps + queue et fréquence NegBin.
    Les paramètres par défaut correspondent au scénario central validé sur PRC 2025.
    """
    # Lognormale du corps calibrée analytiquement sur (médiane, q95)
    mu = np.log(mediane_corps)
    sigma = (np.log(q95_corps) - mu) / stats.norm.ppf(0.95)

    return {
        "source": "PRC 2025 (Volumes) + Jacobs (Euros) + Fréquence NegBin",
        "statut": "Scénario central empirique",
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
    lam = cal["lambda_freq"]
    facteur = cal["facteur_surdispersion"]
    if facteur <= 1.0 + 1e-9:
        n_par_an = rng.poisson(lam, size=M)
    else:
        p_negbin = 1.0 / facteur
        r_negbin = lam * p_negbin / (1.0 - p_negbin)
        n_par_an = rng.negative_binomial(r_negbin, p_negbin, size=M)
    
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

    # 6. Agrégation annuelle ULTRA-RAPIDE (Vectorisation C)
    # Remplace l'ancienne boucle np.split qui ralentissait le bootstrap
    sim_ids = np.repeat(np.arange(M), n_par_an)
    L = np.bincount(sim_ids, weights=sev, minlength=M)

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
    print(" BRIQUE REMEDIATION CORRIGEE — Scénario central + Sensibilité xi effectif")
    print("=" * 72)
    
    # On teste le xi effectif calculé (0.988) ainsi que des sensibilités
    for xi_test in [0.85, 0.988, 1.10]:
        cal = calibrer_remediation(xi=xi_test)
        L = simuler_remediation(cal, M, rng)
        var = np.quantile(L, NIVEAU_VAR)
        print(f"  xi_effectif={xi_test:<5.3f} | VaR99.5 remédiation = {var/1e6:7.2f} M€ "
              f"| moyenne annuelle = {L.mean()/1e6:5.2f} M€")
        
    print("=" * 72)
    print("  Ce test isolé confirme que la brique tourne correctement avec la")
    print("  nouvelle fréquence Binomiale Négative et la calibration PRC/Jacobs.")