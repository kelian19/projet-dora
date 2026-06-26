# -*- coding: utf-8 -*-
"""
=============================================================================
SCR_DORA — SYNTHESE FINALE (Scénario central + Sensibilités + Plausibilité)
=============================================================================

Chaîne complète, calibrée sur la base PRC 2025 réelle, convertie via Jacobs :
  - Remédiation : PRC 2025 + Jacobs (b=0.76) -> xi_effectif ~ 0.988.
                  Fréquence Binomiale Négative (lambda=2.0, facteur=9.20).
  - Sanction    : proxy réglementaire Beta x (2% du CA).
  - Prestataire : scénarios experts (3 quantiles).
  - Aggravation : contrefactuel lognormal.

RESULTAT CENTRAL DE L'ETUDE
---------------------------
À queue lourde (xi_effectif proche de 1), le SCR n'est PAS déterminé par
les données mais par l'HYPOTHESE DE PERTE MAXIMALE (le plafond individuel).
Le capital croît avec le plafond. On ne "mesure" donc pas un SCR ponctuel ;
on le conditionne à un jugement de perte maximale réaliste.

CHOIX CENTRAL : plafond = 40 M€, soit ~5% du chiffre d'affaires de l'entité
(800 M€), niveau cohérent avec la capacité de réassurance cyber mobilisable.
Dépendance de queue : copule de Gumbel, theta = 1.8 (tau de Kendall = 0.44).
"""

import numpy as np

# --- Imports des modules locaux ---
import brique_remediation_corrigee as b_rem
import brique_sanction as b_sanc
import brique_prestataire as b_pres
import brique_aggravation as b_agg
import copule_gumbel_agregation as copule
from ancrage_bilan_plausibilite import profil_assureur, tester_plausibilite_scr, afficher_plausibilite

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

THETA_CENTRAL = 1.8   # tau de Kendall = 1 - 1/theta = 0.44, dépendance de queue de référence
PLAFOND_CENTRAL = 40_000_000.0


def agreger_gumbel(marginales, theta, rng):
    """Agrège les briques par copule de Gumbel (dépendance de queue supérieure)."""
    M_loc = len(marginales[0])
    d = len(marginales)
    if theta <= 1.0 + 1e-9:
        return np.sum(marginales, axis=0)
    U = copule.echantillon_gumbel(M_loc, d, theta, rng)
    L = np.zeros(M_loc)
    for j in range(d):
        L += copule.make_quantile_empirique(marginales[j])(U[:, j])
    return L


def main():
    rng = np.random.default_rng(GRAINE)
    profil = profil_assureur(ca_annuel=800e6)
    tau_central = 1.0 - 1.0 / THETA_CENTRAL   # pour affichage cohérent

    # 1. Calibration
    cal_s = b_sanc.calibrer_sanction(ca_annuel=800e6)
    cal_p = b_pres.calibrer_prestataire()
    cal_a = b_agg.calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2)
    cal_r = b_rem.calibrer_remediation(plafond_individuel=PLAFOND_CENTRAL)

    # 2. Simulation Stand-alone
    L_s = b_sanc.simuler_sanction(cal_s, M, rng)
    L_p = b_pres.simuler_prestataire(cal_p, M, rng)
    L_a = b_agg.simuler_aggravation(cal_a, M, rng)
    L_r = b_rem.simuler_remediation(cal_r, M, rng)

    marginales = [L_r, L_p, L_s, L_a]
    noms = ["Remédiation", "Prestataire", "Sanction", "Aggravation"]

    print("=" * 74)
    print(f" SCR_DORA — SCENARIO CENTRAL (Plafond {PLAFOND_CENTRAL/1e6:.0f} M€, "
          f"Jacobs b=0.76, Theta={THETA_CENTRAL})")
    print("=" * 74)

    for nom, L in zip(noms, marginales):
        print(f"   {nom:14s} VaR99.5 (stand-alone) = {np.quantile(L, NIVEAU_VAR)/1e6:7.2f} M€")

    # Agrégation Centrale (Gumbel) vs Indépendance
    L_dora_gumbel = agreger_gumbel(marginales, THETA_CENTRAL, rng)
    scr_central = np.quantile(L_dora_gumbel, NIVEAU_VAR)

    L_indep = np.sum(marginales, axis=0)
    scr_indep = np.quantile(L_indep, NIVEAU_VAR)

    print(f"   {'-'*60}")
    print(f"   {'L_DORA (Indépendance)':30s} VaR99.5 = {scr_indep/1e6:7.2f} M€")
    print(f"   {f'L_DORA (Gumbel Theta={THETA_CENTRAL}, tau={tau_central:.2f})':30s} "
          f"VaR99.5 = {scr_central/1e6:7.2f} M€  <- REFERENCE")
    print(f"   {'Diversification (Bénéfice)':30s}        = {(scr_central-scr_indep)/1e6:+7.2f} M€ "
          f"({100*(scr_central/scr_indep-1):+.1f}%)")

    print()
    afficher_plausibilite(scr_central, profil)

    # --- Sensibilité au plafond ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE AU PLAFOND — Le capital suit l'hypothèse de perte max !")
    print("=" * 74)
    print(f"   {'Plafond':>10} | {'SCR_DORA':>10} | Verdict Plausibilité")
    print("   " + "-" * 48)
    for pl in [20e6, 40e6, 60e6, 80e6, 100e6]:
        cal_r_temp = b_rem.calibrer_remediation(plafond_individuel=pl)
        L_r_temp = b_rem.simuler_remediation(cal_r_temp, M, rng)
        scr_temp = np.quantile(agreger_gumbel([L_r_temp, L_p, L_s, L_a], THETA_CENTRAL, rng), NIVEAU_VAR)
        t = tester_plausibilite_scr(scr_temp, profil)
        print(f"   {pl/1e6:8.0f}M€ | {scr_temp/1e6:7.1f}M€ | {'PLAUSIBLE' if t['plausible'] else 'NON PLAUSIBLE'}")

    # --- Sensibilité à xi effectif (incertitude EVT x conversion Jacobs) ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE A xi EFFECTIF (indice de queue sur les coûts)")
    print("=" * 74)
    for xi_eff in [0.85, 0.988, 1.10]:
        cal_r_temp = b_rem.calibrer_remediation(xi=xi_eff, plafond_individuel=PLAFOND_CENTRAL)
        L_r_temp = b_rem.simuler_remediation(cal_r_temp, M, rng)
        scr_temp = np.quantile(agreger_gumbel([L_r_temp, L_p, L_s, L_a], THETA_CENTRAL, rng), NIVEAU_VAR)
        print(f"   xi_effectif={xi_eff:<5.3f} -> SCR_DORA = {scr_temp/1e6:6.1f} M€ ")

    print("\n" + "=" * 74)
    print(" CONCLUSION POUR LE MEMOIRE / SOUTENANCE")
    print("=" * 74)
    print(f"""   - A xi proche de 1, l'espérance de la queue diverge. Le SCR est
     gouverné par le plafond (perte max réaliste = {PLAFOND_CENTRAL/1e6:.0f} M€,
     soit ~5% du CA). Il se CONDITIONNE, il ne se mesure pas.
   - Fréquence : Binomiale négative (facteur 9.20) calibrée sur PRC 2025.
   - Dépendance : copule de Gumbel theta={THETA_CENTRAL} (tau={tau_central:.2f}).
   - L'écart avec la formule standard démontre l'apport du modèle interne.""")
    print("=" * 74)


if __name__ == "__main__":
    main()