# -*- coding: utf-8 -*-
"""
=============================================================================
PLAUSIBILITE DU SCR_DORA ALLOUE (et non stand-alone)
=============================================================================

CORRECTION D'UN BIAIS DU TEST DE PLAUSIBILITE
---------------------------------------------
Le test initial confrontait le SCR_DORA *stand-alone* (VaR de la seule perte
DORA) au plafond opérationnel réglementaire 0,3*BSCR. Or ce plafond borne la
charge opérationnelle TOTALE de l'entité (tous risques op confondus), tandis
que le risque DORA n'en est qu'un SOUS-ENSEMBLE (incidents TIC/cyber).
Comparer une partie au plafond du tout surestime le besoin et déclenche de
faux signaux de non-plausibilité dès qu'on injecte la dépendance de queue.

La grandeur réglementairement pertinente est le SCR_DORA *alloué* : la
contribution du risque DORA au capital opérationnel global, obtenue par
allocation d'Euler. L'encadrement du mémoire s'applique :

    SCR_alloué  <=  SCR_DORA  <=  SCR_stand-alone

Ce module calcule le SCR alloué et lui applique le test de plausibilité.

RISQUE OP NON-DORA (calibration)
--------------------------------
Pour allouer, il faut le risque op GLOBAL = DORA + non-DORA. Le non-DORA
(fraude non-cyber, défaut d'exécution des processus, litiges, dommages aux
actifs) est calibré par ANCRAGE sur la formule standard : sa VaR99,5%
stand-alone vise ~75% du SCR_op standard (la part cyber/TIC du risque op est
estimée à 15-30% dans la littérature ORX/Bâle, le complément est non-DORA).
Structure fréquence-sévérité lognormale, SANS queue lourde (xi>1) : ces
risques ne présentent pas l'accumulation systémique propre au cyber.

AVERTISSEMENT : le non-DORA est un proxy de cohérence ancré sur le forfait
réglementaire, pas une calibration empirique sur une base de pertes op réelle.
Le SCR alloué en dépend ; il est donc rapporté avec sa sensibilité à la part
cyber supposée.
"""

import numpy as np
from scipy import stats, optimize

from allocation_euler_var import allocation_euler_var, valider_allocation
from ancrage_bilan_plausibilite import (
    profil_assureur, tester_plausibilite_scr,
)

NIVEAU_VAR = 0.995


# =============================================================================
# RISQUE OPERATIONNEL NON-DORA (proxy ancré sur la formule standard)
# =============================================================================
def calibrer_nondora(profil, part_cyber=0.25, lam=8.0, sigma=1.3, M=1_000_000,
                     graine=7):
    """Calibre un risque op non-DORA dont la VaR99,5% stand-alone vaut
    (1 - part_cyber) * SCR_op_standard.

    part_cyber : part du risque op imputable au cyber/TIC (périmètre DORA).
                 Littérature ORX/Bâle : 15-30%. Défaut 25%.
    lam, sigma : fréquence Poisson et dispersion lognormale (queue légère).
    On résout mu pour atteindre la cible de VaR.
    """
    cible = (1.0 - part_cyber) * profil["scr_op_standard"]
    rng = np.random.default_rng(graine)

    def var_pour_mu(mu):
        n = rng.poisson(lam, size=M)
        tot = int(n.sum())
        sev = rng.lognormal(mu, sigma, size=tot)
        idx = np.cumsum(n)[:-1]
        L = np.array([b.sum() if len(b) > 0 else 0.0 for b in np.split(sev, idx)])
        return np.quantile(L, NIVEAU_VAR)

    mu_opt = optimize.brentq(lambda mu: var_pour_mu(mu) - cible, 10.0, 16.0,
                             xtol=1e-3)
    return {
        "lam": float(lam), "mu": float(mu_opt), "sigma": float(sigma),
        "part_cyber": float(part_cyber), "cible_var": float(cible),
        "source": "Proxy ancré sur SCR_op standard (art.204)",
    }


def simuler_nondora(cal, M, rng):
    """Simule M années de risque op non-DORA (Poisson x Lognormale)."""
    n = rng.poisson(cal["lam"], size=M)
    tot = int(n.sum())
    sev = rng.lognormal(cal["mu"], cal["sigma"], size=tot)
    idx = np.cumsum(n)[:-1]
    return np.array([b.sum() if len(b) > 0 else 0.0 for b in np.split(sev, idx)])


# =============================================================================
# ALLOCATION DU SCR_DORA ET TEST DE PLAUSIBILITE
# =============================================================================
def allouer_et_tester(L_dora, profil, part_cyber=0.25, M=None, graine=123):
    """Calcule le SCR_DORA stand-alone, alloué (Euler) et le SCR op global,
    puis applique le test de plausibilité au SCR ALLOUE.

    L_dora : échantillon Monte-Carlo de la perte DORA agrégée (déjà sous copule).
    """
    M = len(L_dora) if M is None else M
    rng = np.random.default_rng(graine)

    cal_nd = calibrer_nondora(profil, part_cyber=part_cyber)
    L_nondora = simuler_nondora(cal_nd, M, rng)

    scr_standalone = np.quantile(L_dora, NIVEAU_VAR)
    res_alloc = allocation_euler_var([L_dora, L_nondora], niveau=NIVEAU_VAR)
    val = valider_allocation([L_dora, L_nondora], res_alloc, niveau=NIVEAU_VAR)
    scr_alloue = res_alloc["contributions"][0]
    scr_op_global = res_alloc["VaR_globale"]

    test_alloue = tester_plausibilite_scr(scr_alloue, profil)
    test_standalone = tester_plausibilite_scr(scr_standalone, profil)

    return {
        "scr_standalone": scr_standalone,
        "scr_alloue": scr_alloue,
        "scr_op_global": scr_op_global,
        "diversification": scr_standalone - scr_alloue,
        "test_alloue": test_alloue,
        "test_standalone": test_standalone,
        "calibration_nondora": cal_nd,
        "validation_euler": val,
    }


def afficher_rapport(res, profil):
    """Affiche le rapport stand-alone vs alloué."""
    plafond = profil["plafond_op_reglementaire"]
    print("=" * 72)
    print(" PLAUSIBILITE DU SCR_DORA : STAND-ALONE vs ALLOUE (Euler)")
    print("=" * 72)
    print(f"   Plafond op. réglementaire 0,3*BSCR : {plafond/1e6:>8.1f} M€")
    print(f"   SCR op. formule standard           : {profil['scr_op_standard']/1e6:>8.1f} M€")
    print(f"   Risque op non-DORA (proxy ancré)   : "
          f"VaR={res['calibration_nondora']['cible_var']/1e6:.1f} M€ "
          f"(part cyber {res['calibration_nondora']['part_cyber']:.0%})")
    print("   " + "-" * 64)
    print(f"   SCR_DORA stand-alone (borne sup)   : {res['scr_standalone']/1e6:>8.1f} M€  "
          f"-> {'PLAUSIBLE' if res['test_standalone']['plausible'] else 'NON PLAUSIBLE'}")
    print(f"   SCR_DORA ALLOUE (contribution Euler): {res['scr_alloue']/1e6:>8.1f} M€  "
          f"-> {'PLAUSIBLE' if res['test_alloue']['plausible'] else 'NON PLAUSIBLE'}")
    print(f"   SCR op GLOBAL (DORA + non-DORA)    : {res['scr_op_global']/1e6:>8.1f} M€")
    print("   " + "-" * 64)
    print(f"   Bénéfice de diversification        : {res['diversification']/1e6:>8.1f} M€")
    print(f"   Additivité Euler vérifiée          : "
          f"{'OK' if res['validation_euler']['test_additivite'] else 'ECHEC'}")
    print(f"   Diversification (alloué<=standalone): "
          f"{'OK' if res['validation_euler']['test_diversification'] else 'ECHEC'}")
    print("   " + "-" * 64)
    if res["test_alloue"]["plausible"]:
        print("   VERDICT (sur l'alloué) : PLAUSIBLE")
        print("   Le risque DORA stand-alone approche le plafond op total, mais")
        print("   sa contribution réelle au capital (Euler) le respecte. L'écart")
        print("   EST le bénéfice de diversification ignoré par la formule standard.")
    else:
        print("   VERDICT (sur l'alloué) : NON PLAUSIBLE  /!\\")
        for a in res["test_alloue"]["alertes"]:
            print(f"     - {a}")
    print("=" * 72)


def sensibilite_part_cyber(L_dora, profil, parts=(0.15, 0.20, 0.25, 0.30)):
    """Sensibilité du SCR alloué à la part cyber supposée du risque op."""
    print("\n" + "=" * 72)
    print(" SENSIBILITE DU SCR ALLOUE A LA PART CYBER DU RISQUE OP")
    print("=" * 72)
    print(f"   {'part cyber':>11} | {'VaR non-DORA':>13} | {'SCR alloué':>11} | verdict")
    print("   " + "-" * 60)
    for pc in parts:
        r = allouer_et_tester(L_dora, profil, part_cyber=pc)
        v = "PLAUSIBLE" if r["test_alloue"]["plausible"] else "NON PLAUSIBLE"
        print(f"   {pc:>10.0%} | {r['calibration_nondora']['cible_var']/1e6:>11.1f}M€ "
              f"| {r['scr_alloue']/1e6:>9.1f}M€ | {v}")
    print("=" * 72)


# =============================================================================
# DEMONSTRATION AUTONOME
# =============================================================================
if __name__ == "__main__":
    # Reconstruit la perte DORA de référence (copule Gumbel theta=1,8)
    from brique_sanction import calibrer_sanction, simuler_sanction
    from brique_prestataire import calibrer_prestataire, simuler_prestataire
    from brique_aggravation import calibrer_aggravation, simuler_aggravation
    from copule_gumbel_agregation import echantillon_gumbel, make_quantile_empirique
    from scr_dora_synthese_finale import (
        simuler_remediation, PLAFOND_CENTRAL, THETA_GUMBEL,
    )

    M = 500_000
    rng = np.random.default_rng(42)
    profil = profil_assureur(ca_annuel=800e6)

    cal_s = calibrer_sanction(ca_annuel=800e6)
    cal_p = calibrer_prestataire()
    cal_a = calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2)
    L_s = simuler_sanction(cal_s, M, rng)
    L_p = simuler_prestataire(cal_p, M, rng)
    L_a = simuler_aggravation(cal_a, M, rng)
    L_r = simuler_remediation(PLAFOND_CENTRAL, rng=rng)

    U = echantillon_gumbel(M, 4, THETA_GUMBEL, rng)
    margs = [L_s, L_r, L_p, L_a]
    L_dora = np.sum([make_quantile_empirique(margs[j])(U[:, j]) for j in range(4)],
                    axis=0)

    res = allouer_et_tester(L_dora, profil, part_cyber=0.25)
    afficher_rapport(res, profil)
    sensibilite_part_cyber(L_dora, profil)
