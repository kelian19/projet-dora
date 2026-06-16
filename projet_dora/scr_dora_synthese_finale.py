# -*- coding: utf-8 -*-
"""
=============================================================================
SCR_DORA — SYNTHESE FINALE (scénario central + sensibilités + plausibilité)
=============================================================================

Chaîne complète, calibrée sur la base PRC réelle (incidents HACK comme proxy
du périmètre DORA, approche A) :

  - Remédiation : corps lognormal (médiane 171 k€, q95 9,3 M€ issus de la
    conversion records->EUR sur les HACK réels) + queue GPD (xi littérature
    [1,1;1,5]) au-dessus du seuil POT (q90 = 4,95 M€), plafond individuel ancré.
  - Sanction    : proxy réglementaire Beta x (2% du CA).
  - Prestataire : scénarios experts (3 quantiles).
  - Aggravation : contrefactuel lognormal.

RESULTAT CENTRAL DE L'ETUDE
---------------------------
A queue lourde (xi>1), le SCR n'est PAS déterminé par les données mais par
l'HYPOTHESE DE PERTE MAXIMALE (le plafond individuel). La courbe SCR(plafond)
le démontre : le capital croît linéairement avec le plafond. On ne "mesure"
donc pas un SCR ponctuel ; on le conditionne à un jugement de perte maximale
réaliste. C'est la traduction quantitative de la Remarque 3 du document.

CHOIX CENTRAL : plafond = 50 M€, ancré sur la capacité de réassurance cyber
disponible (~10% des fonds propres de l'entité). Donne un SCR plausible
(~76 M€) qui reste 2,4x la charge de la formule standard -> l'argument
d'insensibilité de la formule standard au profil TIC tient.

VALIDATIONS AUTOMATIQUES (garde-fous) :
  - plausibilité réglementaire : SCR <= 0,3*BSCR, < CA, < fonds propres
  - sensibilité à xi et au plafond rapportée en fourchette, pas en point

AVERTISSEMENT : calibration partiellement à dire d'expert (xi, plafond,
fréquence-entité). Le SCR est conditionnel aux hypothèses, présenté en
fourchette. Voir le document méthodologique, section sensibilité.
"""

import numpy as np
from scipy import stats

from brique_sanction import calibrer_sanction, simuler_sanction
from brique_prestataire import calibrer_prestataire, simuler_prestataire
from brique_aggravation import calibrer_aggravation, simuler_aggravation
from ancrage_bilan_plausibilite import (
    profil_assureur, tester_plausibilite_scr, afficher_plausibilite,
)

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

# --- Paramètres remédiation issus de la conversion PRC (HACK réels) ---
REM_MEDIANE_CORPS = 171_000.0
REM_Q95_CORPS = 9_340_000.0
REM_SEUIL_POT = 4_950_000.0
REM_BETA = 2_000_000.0
REM_P_QUEUE = 0.10
REM_LAMBDA_ENTITE = 2.0   # fréquence ramenée à UNE entité (cf. note ci-dessous)
REM_XI_CENTRAL = 1.3      # littérature cyber, testé en sensibilité

# --- Plafond central ancré (réassurance cyber ~10% fonds propres) ---
PLAFOND_CENTRAL = 50_000_000.0

# Note fréquence : la PRC recense 126 incidents HACK sur ~10 ans pour TOUT un
# marché (multi-entités). Rapportée à UNE entité d'assurance, la fréquence
# propre est bien plus faible. On retient lambda=2/an comme hypothèse entité,
# à documenter et tester. Utiliser 12,6/an (marché entier) surestimerait.


def _params_corps():
    mu = np.log(REM_MEDIANE_CORPS)
    sigma = (np.log(REM_Q95_CORPS) - mu) / stats.norm.ppf(0.95)
    return mu, sigma


def simuler_remediation(plafond, lam=REM_LAMBDA_ENTITE, xi=REM_XI_CENTRAL,
                        beta=REM_BETA, p_queue=REM_P_QUEUE, rng=None):
    """Remédiation corps+queue, plafond ancré. Voir docstring module."""
    mu, sigma = _params_corps()
    n = rng.poisson(lam, size=M)
    tot = int(n.sum())
    en_q = rng.random(tot) < p_queue
    sev = np.empty(tot)
    nq = int(en_q.sum())
    if nq > 0:
        sev[en_q] = REM_SEUIL_POT + stats.genpareto.rvs(
            c=xi, scale=beta, size=nq, random_state=rng)
    if tot - nq > 0:
        sev[~en_q] = rng.lognormal(mu, sigma, size=tot - nq)
    sev = np.minimum(sev, plafond)
    idx = np.cumsum(n)[:-1]
    return np.array([b.sum() if len(b) > 0 else 0.0 for b in np.split(sev, idx)])


def construire_briques(rng):
    cal_s = calibrer_sanction(ca_annuel=800e6)
    cal_p = calibrer_prestataire()
    cal_a = calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2)
    L_s = simuler_sanction(cal_s, M, rng)
    L_p = simuler_prestataire(cal_p, M, rng)
    L_a = simuler_aggravation(cal_a, M, rng)
    return L_s, L_p, L_a


def main():
    rng = np.random.default_rng(GRAINE)
    profil = profil_assureur(ca_annuel=800e6)
    L_s, L_p, L_a = construire_briques(rng)

    print("=" * 74)
    print(" SCR_DORA — SCENARIO CENTRAL (plafond 50 M€, xi=1,3)")
    print("=" * 74)
    L_r = simuler_remediation(PLAFOND_CENTRAL, rng=rng)
    L_dora = L_s + L_r + L_p + L_a
    for nom, L in [("Sanction", L_s), ("Remédiation", L_r),
                   ("Prestataire", L_p), ("Aggravation", L_a)]:
        print(f"   {nom:14s} VaR99.5 = {np.quantile(L, NIVEAU_VAR)/1e6:7.2f} M€")
    scr_central = np.quantile(L_dora, NIVEAU_VAR)
    print(f"   {'L_DORA total':14s} VaR99.5 = {scr_central/1e6:7.2f} M€")

    print()
    afficher_plausibilite(scr_central, profil)

    # --- Sensibilité à xi (fourchette, pas un point) ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE A xi (plafond central 50 M€)")
    print("=" * 74)
    for xi in [1.1, 1.3, 1.5]:
        L_r = simuler_remediation(PLAFOND_CENTRAL, xi=xi, rng=rng)
        scr = np.quantile(L_s + L_r + L_p + L_a, NIVEAU_VAR)
        print(f"   xi={xi:.1f} -> SCR_DORA = {scr/1e6:6.1f} M€ "
              f"({scr/profil['scr_op_standard']:.1f}x formule standard)")

    # --- Sensibilité au plafond (LE résultat-clé) ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE AU PLAFOND — le capital suit l'hypothèse de perte max")
    print("=" * 74)
    print(f"   {'plafond':>10} | {'SCR_DORA':>10} | verdict plausibilité")
    print("   " + "-" * 48)
    for pl in [20e6, 50e6, 75e6, 100e6, 196e6]:
        L_r = simuler_remediation(pl, rng=rng)
        scr = np.quantile(L_s + L_r + L_p + L_a, NIVEAU_VAR)
        t = tester_plausibilite_scr(scr, profil)
        print(f"   {pl/1e6:8.0f}M€ | {scr/1e6:7.1f}M€ | "
              f"{'PLAUSIBLE' if t['plausible'] else 'NON PLAUSIBLE'}")

    print("\n" + "=" * 74)
    print(" CONCLUSION POUR LE MEMOIRE")
    print("=" * 74)
    print("""   - A xi>1, le SCR est gouverné par le plafond (perte max réaliste),
     pas par les données : il se CONDITIONNE, il ne se mesure pas.
   - Scénario central retenu : plafond 50 M€ -> SCR ~76 M€, PLAUSIBLE,
     soit ~2,4x la charge de la formule standard (32 M€).
   - L'écart au forfait standard démontre son insensibilité au profil TIC :
     c'est l'apport du modèle interne, indépendamment de la valeur exacte.
   - Le SCR est présenté en FOURCHETTE (xi, plafond), pas en point unique.""")
    print("=" * 74)


if __name__ == "__main__":
    main()
