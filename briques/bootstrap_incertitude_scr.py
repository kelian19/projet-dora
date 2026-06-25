# -*- coding: utf-8 -*-
"""
=============================================================================
BOOTSTRAP PARAMETRIQUE — PROPAGATION DE L'INCERTITUDE JUSQU'AU SCR
=============================================================================
Ce module propage les incertitudes de calibration (GPD, Jacobs, experts) 
jusqu'au SCR_DORA via un Monte-Carlo à deux niveaux (Bootstrap).
"""

import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from numpy.linalg import lstsq

import brique_sanction as b_sanc
import brique_prestataire as b_pres
import brique_aggravation as b_agg
import copule_gumbel_agregation as copule
from ancrage_bilan_plausibilite import profil_assureur

NIVEAU_VAR = 0.995
B = 500            # réplications de paramètres (niveau externe)
M_INT = 100_000    # simulations par réplication (niveau interne)
GRAINE = 2026
PLAFOND = 40_000_000.0   # Plafond ancré (fixé)

def tirer_parametres(rng):
    """Tire un jeu de paramètres dans leurs lois d'incertitude respectives."""
    # --- Remédiation (Incertitude sur la conversion Jacobs et la queue EVT) ---
    xi_eff = rng.uniform(0.85, 1.10)   # Combine incertitude xi_vol et pente Jacobs
    
    cv = 0.30
    k = 1.0 / cv**2
    lam = rng.gamma(k, 2.0 / k)        # Fréquence entité incertaine
    facteur = rng.uniform(7.0, 11.0)   # Autour du 9.2 estimé
    theta = rng.uniform(1.2, 1.8)      # Incertitude sur la dépendance de Gumbel

    # --- Sanction ---
    alpha_s = rng.uniform(0.5, 1.5)
    beta_s  = rng.uniform(4.0, 9.0)
    proba_s = rng.uniform(0.05, 0.20)

    # --- Prestataire : incertitude ±30% sur les quantiles experts ---
    scenarios_ref = [
        {"nom": "Panne cloud", "proba_an": 0.08, "q50": 2e6, "q95": 25e6, "q995": 150e6, "loi": "gpd"},
        {"nom": "Compromission paiement", "proba_an": 0.05, "q50": 1e6, "q95": 12e6, "q995": 80e6, "loi": "gpd"},
        {"nom": "Ransomware éditeur", "proba_an": 0.12, "q50": 3e5, "q95": 3e6, "q995": 20e6, "loi": "lognorm"},
    ]
    scenarios_tires = []
    for s in scenarios_ref:
        s_tire = dict(s)
        s_tire["q95"]  = max(s["q95"] * np.exp(rng.normal(0.0, 0.30)), s["q50"] * 1.01)
        s_tire["q995"] = max(s["q995"] * np.exp(rng.normal(0.0, 0.30)), s_tire["q95"] * 1.01)
        scenarios_tires.append(s_tire)

    return {
        "xi_eff": xi_eff, "lam": lam, "facteur": facteur, "theta": theta,
        "alpha_s": alpha_s, "beta_s": beta_s, "proba_s": proba_s,
        "scenarios_presta": scenarios_tires,
    }

def simuler_remediation_param(p, rng):
    """Simule la brique remédiation avec vectorisation ultra-rapide."""
    # Paramètres de base du scénario central en euros
    mu_corps, sigma_corps = np.log(603_000.0), 1.9
    seuil_u, beta_gpd, p_queue = 15_310_000.0, 21_000_000.0, 0.0654

    # Tirage fréquence NegBin
    if p["facteur"] <= 1.0:
        n = rng.poisson(p["lam"], size=M_INT)
    else:
        pn = 1.0 / p["facteur"]
        rn = p["lam"] * pn / (1.0 - pn)
        n = rng.negative_binomial(rn, pn, size=M_INT)
        
    tot = int(n.sum())
    if tot == 0: return np.zeros(M_INT)
    
    en_q = rng.random(tot) < p_queue
    sev = np.empty(tot)
    nq = int(en_q.sum())
    
    if nq > 0:
        sev[en_q] = seuil_u + stats.genpareto.rvs(c=p["xi_eff"], scale=beta_gpd, size=nq, random_state=rng)
    if tot - nq > 0:
        sev[~en_q] = rng.lognormal(mu_corps, sigma_corps, size=tot - nq)
        
    sev = np.minimum(sev, PLAFOND)
    
    # Vectorisation C (bincount)
    sim_ids = np.repeat(np.arange(M_INT), n)
    return np.bincount(sim_ids, weights=sev, minlength=M_INT)

def scr_une_replication(p, rng):
    cal_s = b_sanc.calibrer_sanction(ca_annuel=800e6, alpha_defaut=p["alpha_s"], beta_defaut=p["beta_s"], proba_survenance=p["proba_s"])
    cal_p = b_pres.calibrer_prestataire(scenarios=p["scenarios_presta"])
    cal_a = b_agg.calibrer_aggravation()

    L_s = b_sanc.simuler_sanction(cal_s, M_INT, rng)
    L_p = b_pres.simuler_prestataire(cal_p, M_INT, rng)
    L_a = b_agg.simuler_aggravation(cal_a, M_INT, rng)
    L_r = simuler_remediation_param(p, rng)
    
    U = copule.echantillon_gumbel(M_INT, 4, p["theta"], rng)
    marginales = [L_s, L_r, L_p, L_a]
    L = np.sum([copule.make_quantile_empirique(marginales[j])(U[:, j]) for j in range(4)], axis=0)
    return np.quantile(L, NIVEAU_VAR)

def main():
    print("=" * 70)
    print(" BOOTSTRAP PARAMETRIQUE DU SCR_DORA (Génération de la distribution)")
    print(f" {B} réplications x {M_INT:,} simulations")
    print("=" * 70)
    
    rng = np.random.default_rng(GRAINE)
    scrs, params = np.empty(B), []
    
    for b in range(B):
        p = tirer_parametres(rng)
        scrs[b] = scr_une_replication(p, rng)
        params.append(p)
        if (b + 1) % 50 == 0:
            print(f"   ... {b+1}/{B} réplications  (SCR médian = {np.median(scrs[:b+1])/1e6:.1f} M€)")

    print("\n" + "=" * 70)
    print(" DISTRIBUTION DU SCR_DORA SOUS INCERTITUDE PARAMETRIQUE")
    print("=" * 70)
    print(f"   Médiane                    : {np.median(scrs)/1e6:7.1f} M€")
    print(f"   Moyenne                    : {scrs.mean()/1e6:7.1f} M€")
    print(f"   IC 90% [q05 ; q95]         : [{np.quantile(scrs, 0.05)/1e6:.1f} ; {np.quantile(scrs, 0.95)/1e6:.1f}] M€")
    
    # Décomposition de variance
    keys = ["xi_eff", "lam", "facteur", "theta", "alpha_s", "beta_s", "proba_s"]
    mat = {k: np.array([p[k] for p in params]) for k in keys}
    q95_presta = np.array([np.mean([s["q95"] for s in p["scenarios_presta"]]) for p in params])
    mat["q95_presta"] = q95_presta
    keys.append("q95_presta")
    
    X = np.column_stack([(mat[k] - mat[k].mean()) / mat[k].std() for k in keys])
    y = (scrs - scrs.mean()) / scrs.std()
    beta, *_ = lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)
    b2 = beta[1:] ** 2
    part = b2 / b2.sum()
    
    print("\n   QUI PILOTE L'INCERTITUDE DU SCR ? (Décomposition de variance)")
    print("   " + "-" * 60)
    for k, p_var in sorted(zip(keys, part), key=lambda x: -x[1]):
        print(f"   {k:20s} : {p_var:>7.1%}")
    print("=" * 70)

if __name__ == "__main__":
    main()