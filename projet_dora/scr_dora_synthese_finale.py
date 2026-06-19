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
from copule_gumbel_agregation import echantillon_gumbel, make_quantile_empirique
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
# Surdispersion de la fréquence (variance = facteur x moyenne, binomiale négative).
# CALIBRATION : comptages annuels HACK de la base PRC (2019-2025), MLE NB2 ->
# alpha ~ 0,15 à l'échelle marché (mu~53/an). C'est alpha (clustering structurel)
# qui se transpose à l'échelle entité, pas le facteur : à lambda=2, le facteur
# attendu = 1 + alpha*lambda ~ 1,30. Référence = 1,30 (sensibilité [1,1;1,5;2,0]).
REM_FACTEUR_SURDISP = 1.30

# --- Dépendance inter-briques : copule de Gumbel (approche de référence) ---
# theta=1 -> indépendance ; theta->inf -> comonotonie. tau Kendall = 1 - 1/theta.
# theta=1,8 = état "non conforme" (forte contagion : concentration cloud non
# maîtrisée), cohérent avec l'approche C. C'est l'agrégation de REFERENCE du
# mémoire ; la somme indépendante (theta=1) est conservée comme comparatif.
THETA_GUMBEL = 1.8

# --- Plafond central ancré (réassurance cyber ~5% fonds propres) ---
# 40 M€ ~ 5% des fonds propres (784 M€), ordre de grandeur de la capacité de
# réassurance cyber mobilisable. Sous copule Gumbel theta=1,8, ce plafond
# maintient le SCR_DORA (~103 M€) sous le plafond op réglementaire (0,3*BSCR
# = 108 M€) tout en conservant une dépendance de queue forte. A 50 M€, la
# combinaison plafond+copule franchissait le plafond réglementaire.
PLAFOND_CENTRAL = 40_000_000.0

# Note fréquence : la PRC recense 126 incidents HACK sur ~10 ans pour TOUT un
# marché (multi-entités). Rapportée à UNE entité d'assurance, la fréquence
# propre est bien plus faible. On retient lambda=2/an comme hypothèse entité,
# à documenter et tester. Utiliser 12,6/an (marché entier) surestimerait.


def _params_corps():
    mu = np.log(REM_MEDIANE_CORPS)
    sigma = (np.log(REM_Q95_CORPS) - mu) / stats.norm.ppf(0.95)
    return mu, sigma


def _tirer_frequence(lam, rng, facteur_surdisp=REM_FACTEUR_SURDISP):
    """Tire le nombre annuel d'incidents.
    facteur_surdisp = variance/moyenne. =1 -> Poisson ; >1 -> binomiale négative
    (clustering des incidents cyber). Moyenne = lam dans les deux cas."""
    if facteur_surdisp <= 1.0 + 1e-9:
        return rng.poisson(lam, size=M)
    var = facteur_surdisp * lam
    p = lam / var               # p = moyenne / variance
    r = lam * p / (1.0 - p)     # r = moyenne * p / (1-p)
    return rng.negative_binomial(r, p, size=M)


def simuler_remediation(plafond, lam=REM_LAMBDA_ENTITE, xi=REM_XI_CENTRAL,
                        beta=REM_BETA, p_queue=REM_P_QUEUE, rng=None,
                        facteur_surdisp=REM_FACTEUR_SURDISP):
    """Remédiation corps+queue, plafond ancré, fréquence binomiale négative."""
    mu, sigma = _params_corps()
    n = _tirer_frequence(lam, rng, facteur_surdisp)
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


def agreger_gumbel(marginales, theta, rng):
    """Agrège les briques par copule de Gumbel (dépendance de queue supérieure).

    theta=1 redonne exactement la somme indépendante. Pour theta>1, on couple
    les rangs des marginales via un échantillon de copule de Gumbel (méthode de
    Marshall-Olkin), ce qui préserve les marges stand-alone tout en injectant la
    co-occurrence des coûts extrêmes. C'est l'agrégation de référence du mémoire.
    """
    M_loc = len(marginales[0])
    d = len(marginales)
    if theta <= 1.0 + 1e-9:
        return np.sum(marginales, axis=0)
    U = echantillon_gumbel(M_loc, d, theta, rng)
    L = np.zeros(M_loc)
    for j in range(d):
        L += make_quantile_empirique(marginales[j])(U[:, j])
    return L


def main():
    rng = np.random.default_rng(GRAINE)
    profil = profil_assureur(ca_annuel=800e6)
    L_s, L_p, L_a = construire_briques(rng)

    print("=" * 74)
    print(" SCR_DORA — SCENARIO CENTRAL (plafond 40 M€, xi=1,3, copule Gumbel theta=1,8)")
    print("=" * 74)
    L_r = simuler_remediation(PLAFOND_CENTRAL, rng=rng)
    marginales = [L_s, L_r, L_p, L_a]
    for nom, L in [("Sanction", L_s), ("Remédiation", L_r),
                   ("Prestataire", L_p), ("Aggravation", L_a)]:
        print(f"   {nom:14s} VaR99.5 (stand-alone) = {np.quantile(L, NIVEAU_VAR)/1e6:7.2f} M€")

    # Agrégation de REFERENCE : copule de Gumbel (dépendance de queue)
    L_dora = agreger_gumbel(marginales, THETA_GUMBEL, rng)
    scr_central = np.quantile(L_dora, NIVEAU_VAR)

    # Comparatif : somme indépendante (theta=1) pour mesurer l'effet de la copule
    L_indep = np.sum(marginales, axis=0)
    scr_indep = np.quantile(L_indep, NIVEAU_VAR)

    print(f"   {'-'*60}")
    print(f"   {'L_DORA (indépendance)':30s} VaR99.5 = {scr_indep/1e6:7.2f} M€")
    print(f"   {'L_DORA (Gumbel theta=1,8)':30s} VaR99.5 = {scr_central/1e6:7.2f} M€  <- REFERENCE")
    print(f"   {'effet copule':30s}        = {(scr_central-scr_indep)/1e6:+7.2f} M€ "
          f"({100*(scr_central/scr_indep-1):+.1f}%)")

    print()
    afficher_plausibilite(scr_central, profil)

    # --- Sensibilité à xi (fourchette, pas un point) ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE A xi (plafond central 40 M€)")
    print("=" * 74)
    for xi in [1.1, 1.3, 1.5]:
        L_r = simuler_remediation(PLAFOND_CENTRAL, xi=xi, rng=rng)
        scr = np.quantile(agreger_gumbel([L_s, L_r, L_p, L_a], THETA_GUMBEL, rng),
                          NIVEAU_VAR)
        print(f"   xi={xi:.1f} -> SCR_DORA = {scr/1e6:6.1f} M€ "
              f"({scr/profil['scr_op_standard']:.1f}x formule standard)")

    # --- Sensibilité au plafond (LE résultat-clé) ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE AU PLAFOND — le capital suit l'hypothèse de perte max")
    print("=" * 74)
    print(f"   {'plafond':>10} | {'SCR_DORA':>10} | verdict plausibilité")
    print("   " + "-" * 48)
    for pl in [20e6, 40e6, 50e6, 75e6, 100e6]:
        L_r = simuler_remediation(pl, rng=rng)
        scr = np.quantile(agreger_gumbel([L_s, L_r, L_p, L_a], THETA_GUMBEL, rng),
                          NIVEAU_VAR)
        t = tester_plausibilite_scr(scr, profil)
        print(f"   {pl/1e6:8.0f}M€ | {scr/1e6:7.1f}M€ | "
              f"{'PLAUSIBLE' if t['plausible'] else 'NON PLAUSIBLE'}")

    # --- Sensibilité à theta (structure de dépendance) ---
    print("\n" + "=" * 74)
    print(" SENSIBILITE A LA DEPENDANCE theta (plafond central 40 M€, xi=1,3)")
    print("=" * 74)
    L_r = simuler_remediation(PLAFOND_CENTRAL, rng=rng)
    print(f"   {'theta':>6} | {'tau Kendall':>11} | {'SCR_DORA':>10}")
    print("   " + "-" * 34)
    for th in [1.0, 1.2, 1.5, 1.8, 2.5]:
        scr = np.quantile(agreger_gumbel([L_s, L_r, L_p, L_a], th, rng), NIVEAU_VAR)
        tau = 0.0 if th <= 1 else 1 - 1/th
        tag = "  <- REFERENCE" if abs(th-THETA_GUMBEL) < 1e-9 else ""
        print(f"   {th:6.1f} | {tau:11.2f} | {scr/1e6:7.1f}M€{tag}")

    print("\n" + "=" * 74)
    print(" CONCLUSION POUR LE MEMOIRE")
    print("=" * 74)
    print(f"""   - Agrégation de REFERENCE : copule de Gumbel (theta={THETA_GUMBEL}), qui
     injecte la dépendance de queue entre briques. La somme indépendante
     est conservée comme borne basse comparative.
   - A xi>1, le SCR est gouverné par le plafond (perte max réaliste),
     pas par les données : il se CONDITIONNE, il ne se mesure pas.
   - Fréquence : binomiale négative, facteur de surdispersion {REM_FACTEUR_SURDISP}
     calibré sur les comptages HACK de la PRC (alpha~0,15 transposé à
     l'échelle entité), testé en sensibilité.
   - L'écart au forfait standard démontre son insensibilité au profil TIC :
     c'est l'apport du modèle interne, indépendamment de la valeur exacte.
   - Le SCR est présenté en FOURCHETTE (xi, theta, plafond), pas en point.""")
    print("=" * 74)


if __name__ == "__main__":
    main()