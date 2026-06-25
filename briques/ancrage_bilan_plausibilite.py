# -*- coding: utf-8 -*-
"""
=============================================================================
ANCRAGE BILANCIEL & TEST DE PLAUSIBILITE REGLEMENTAIRE
=============================================================================

Corrige le défaut majeur du modèle : un SCR_DORA de ~2000 M€ pour un assureur
à 800 M€ de CA est économiquement impossible. La cause : un plafond de
sévérité individuel (500 M€/sinistre) sans ancrage bilanciel, qui, combiné à
xi>1, laisse la VaR exploser.

Ce module fournit :
  1. Un profil bilanciel d'assureur synthétique cohérent (CA, BSCR, fonds
     propres) servant à ANCRER le plafond de sévérité.
  2. Un plafond de sévérité individuel dérivé du bilan (perte max réaliste).
  3. Un TEST DE PLAUSIBILITE automatique du SCR_DORA contre deux bornes
     réglementaires :
       - le plafond dur de la formule standard : SCR_op <= 0,30 * BSCR
         (art. 204 du Règlement délégué 2015/35)
       - la charge forfaitaire de la formule standard elle-même
     Si le SCR_DORA dépasse largement ces repères, le modèle est signalé
     comme non plausible (ce qui aurait stoppé les 2000 M€).

REFERENCES : art. 204 Reg. délégué 2015/35 (SCR_op = min(0,3*BSCR ; Op)
+ 0,25*Exp_ul) ; ordre de grandeur SCR/fonds propres d'un assureur réel.

AVERTISSEMENT : profil bilanciel synthétique. Sert d'ancrage de cohérence,
pas de calibration empirique.
"""

import numpy as np


# =============================================================================
# 1. PROFIL BILANCIEL DE L'ASSUREUR SYNTHETIQUE
# =============================================================================
def profil_assureur(
    ca_annuel=800_000_000.0,
    ratio_bscr_ca=0.45,          # BSCR ~ 45% du CA (ordre de grandeur assureur)
    ratio_fonds_propres_scr=2.0, # fonds propres ~ 2x le SCR (couverture ~200%)
    primes_brutes=None,
    provisions_techniques=None,
):
    """
    Construit un profil bilanciel cohérent pour ancrer le modèle.

    Les ratios par défaut reflètent des ordres de grandeur observés sous
    Solvabilité 2 (couverture SCR ~150-250%, BSCR fraction du volume d'activité).

    Retour : dict avec CA, BSCR, SCR_global approx, fonds propres, et les
    bornes réglementaires utiles au test de plausibilité.
    """
    if primes_brutes is None:
        primes_brutes = ca_annuel            # proxy : CA ~ primes
    if provisions_techniques is None:
        provisions_techniques = 1.5 * ca_annuel

    bscr = ratio_bscr_ca * ca_annuel
    # SCR global approx = BSCR + SCR_op (on prend SCR_op standard ci-dessous)
    op_primes = 0.04 * primes_brutes         # art.204 : ~3-4% des primes
    op_provisions = 0.0045 * provisions_techniques
    op_standard = max(op_primes, op_provisions)
    scr_op_standard = min(op_standard, 0.30 * bscr)   # formule standard art.204

    scr_global = bscr + scr_op_standard
    fonds_propres = ratio_fonds_propres_scr * scr_global

    return {
        "ca_annuel": ca_annuel,
        "primes_brutes": primes_brutes,
        "provisions_techniques": provisions_techniques,
        "bscr": bscr,
        "scr_op_standard": scr_op_standard,     # charge forfaitaire art.204
        "plafond_op_reglementaire": 0.30 * bscr, # plafond dur SCR_op
        "scr_global": scr_global,
        "fonds_propres": fonds_propres,
    }


# =============================================================================
# 2. PLAFOND DE SEVERITE INDIVIDUEL ANCRE SUR LE BILAN
# =============================================================================
def plafond_severite_individuel(profil, fraction_fonds_propres=0.25):
    """
    Perte maximale réaliste d'UN sinistre, ancrée sur le bilan.

    Principe : un sinistre opérationnel unique ne peut raisonnablement pas
    excéder une fraction des fonds propres de l'entité sans la mettre en
    ruine immédiate. On retient par défaut 25% des fonds propres comme
    perte individuelle maximale réaliste (au-delà, l'entité fait défaut et
    le cadre going-concern ne s'applique plus).

    Pour un assureur à 800 M€ de CA -> fonds propres ~ 700 M€ -> plafond
    individuel ~ 175 M€, très inférieur aux 500 M€ arbitraires précédents.
    """
    return fraction_fonds_propres * profil["fonds_propres"]


# =============================================================================
# 3. TEST DE PLAUSIBILITE DU SCR_DORA
# =============================================================================
def tester_plausibilite_scr(scr_dora, profil, tolerance=1.0):
    """
    Confronte le SCR_DORA aux repères réglementaires. Le risque DORA n'est
    qu'une PARTIE du risque opérationnel ; son SCR ne devrait donc pas
    dépasser le plafond opérationnel réglementaire (0,3*BSCR), et devrait
    rester du même ordre que la charge opérationnelle standard.

    tolerance : multiple du plafond réglementaire au-delà duquel on alerte.

    Retour : dict avec verdict et ratios explicatifs.
    """
    plafond_op = profil["plafond_op_reglementaire"]
    scr_op_std = profil["scr_op_standard"]
    ca = profil["ca_annuel"]
    fp = profil["fonds_propres"]

    ratio_vs_plafond = scr_dora / plafond_op
    ratio_vs_standard = scr_dora / scr_op_std
    ratio_vs_ca = scr_dora / ca
    ratio_vs_fp = scr_dora / fp

    # Critères de plausibilité
    alertes = []
    if scr_dora > tolerance * plafond_op:
        alertes.append(
            f"SCR_DORA ({scr_dora/1e6:.0f} M€) > plafond op. réglementaire "
            f"0,3*BSCR ({plafond_op/1e6:.0f} M€). Le risque DORA seul dépasse "
            f"le plafond de TOUT le risque opérationnel : non plausible.")
    if ratio_vs_ca > 1.0:
        alertes.append(
            f"SCR_DORA dépasse le chiffre d'affaires annuel "
            f"({ratio_vs_ca:.1f}x le CA) : non plausible.")
    if ratio_vs_fp > 1.0:
        alertes.append(
            f"SCR_DORA dépasse les fonds propres "
            f"({ratio_vs_fp:.1f}x) : l'entité serait en ruine.")

    plausible = len(alertes) == 0

    return {
        "plausible": plausible,
        "alertes": alertes,
        "ratio_vs_plafond_op": ratio_vs_plafond,
        "ratio_vs_standard": ratio_vs_standard,
        "ratio_vs_ca": ratio_vs_ca,
        "ratio_vs_fonds_propres": ratio_vs_fp,
        "plafond_op_reglementaire": plafond_op,
        "scr_op_standard": scr_op_std,
    }


def afficher_plausibilite(scr_dora, profil):
    """Affiche le rapport de plausibilité de façon lisible."""
    res = tester_plausibilite_scr(scr_dora, profil)
    print("=" * 72)
    print(" TEST DE PLAUSIBILITE REGLEMENTAIRE DU SCR_DORA")
    print("=" * 72)
    print(f"   SCR_DORA évalué            : {scr_dora/1e6:>10.1f} M€")
    print(f"   Chiffre d'affaires         : {profil['ca_annuel']/1e6:>10.1f} M€")
    print(f"   BSCR (approx)              : {profil['bscr']/1e6:>10.1f} M€")
    print(f"   Plafond op. (0,3*BSCR)     : {profil['plafond_op_reglementaire']/1e6:>10.1f} M€")
    print(f"   SCR op. formule standard   : {profil['scr_op_standard']/1e6:>10.1f} M€")
    print(f"   Fonds propres (approx)     : {profil['fonds_propres']/1e6:>10.1f} M€")
    print("   " + "-" * 64)
    print(f"   SCR_DORA / plafond op.     : {res['ratio_vs_plafond_op']:>10.2f}")
    print(f"   SCR_DORA / SCR op. std     : {res['ratio_vs_standard']:>10.2f}")
    print(f"   SCR_DORA / CA              : {res['ratio_vs_ca']:>10.2f}")
    print("   " + "-" * 64)
    if res["plausible"]:
        print("   VERDICT : PLAUSIBLE")
        print("   (le SCR_DORA reste dans les bornes réglementaires)")
    else:
        print("   VERDICT : NON PLAUSIBLE  /!\\")
        for a in res["alertes"]:
            print(f"     - {a}")
    print("=" * 72)
    return res


# =============================================================================
# DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    profil = profil_assureur(ca_annuel=800_000_000.0)

    print("=" * 72)
    print(" PROFIL BILANCIEL DE L'ASSUREUR SYNTHETIQUE")
    print("=" * 72)
    for k, v in profil.items():
        print(f"   {k:28s}: {v/1e6:>12.1f} M€")

    plafond = plafond_severite_individuel(profil)
    print(f"\n   => Plafond de sévérité individuel ancré : {plafond/1e6:.1f} M€")
    print(f"      (= 25% des fonds propres, vs 500 M€ arbitraire précédent)")

    # Démonstration du test sur les deux cas : ancien (absurde) vs cible
    print("\n\n--- CAS 1 : ancien résultat (plafond 500 M€, non ancré) ---")
    afficher_plausibilite(2008e6, profil)

    print("\n\n--- CAS 2 : ordre de grandeur cible (plafond ancré) ---")
    afficher_plausibilite(45e6, profil)
