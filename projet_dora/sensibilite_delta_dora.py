# -*- coding: utf-8 -*-
"""
=============================================================================
SENSIBILITE DE Delta_DORA AUX HYPOTHESES D'EFFICACITE DE LA CONFORMITE
=============================================================================

L'approche C (contrefactuel) chiffre le cout du non-respect comme :
    Delta_DORA = VaR(non conforme) - VaR(conforme)
Mais ce chiffre depend ENTIEREMENT de deux hypotheses posees a dire d'expert :
  - le taux de reduction de FREQUENCE apporte par la conformite (hygiene cyber)
  - l'ecart de DEPENDANCE de queue theta (stratEgies de sortie / contagion)

Plutot que de presenter Delta_DORA comme un point unique (29 M€), ce module
construit une GRILLE : Delta_DORA pour plusieurs jeux d'hypotheses. On lit
alors le cout du non-respect comme une FOURCHETTE conditionnelle, ce qui
reflete honnetement l'incertitude et repond a la critique "vos ratios sont
poses".

POURQUOI PAS PSM / IPW / diff-in-diff : ces methodes d'inference causale
exigent des donnees reelles conformes/non-conformes, qui n'existent pas ici.
Les appliquer sur donnees simulees serait circulaire (on ne retrouverait que
l'effet injecte). La sensibilite aux hypotheses est le test honnete possible.

AVERTISSEMENT : meme protocole que approche_c (graine 42, M=5e5, profil
constant). Le profil sous-jacent est identique des deux cotes : seuls le
ratio de frequence et theta varient -> biais de selection neutralise.
"""

import numpy as np
from scipy import stats

from brique_sanction import calibrer_sanction, simuler_sanction
from brique_prestataire import calibrer_prestataire, simuler_prestataire
from brique_aggravation import calibrer_aggravation, simuler_aggravation
from copule_gumbel_agregation import echantillon_gumbel, make_quantile_empirique
from ancrage_bilan_plausibilite import profil_assureur, tester_plausibilite_scr

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

# Parametres remediation (PRC HACK reels, cf. synthese)
REM_MEDIANE = 171_000.0
REM_Q95 = 9_340_000.0
REM_SEUIL_POT = 4_950_000.0
REM_BETA = 2_000_000.0
REM_P_QUEUE = 0.10
REM_XI = 1.3
PLAFOND_INDIV = 40_000_000.0   # plafond central 40 M euros, aligne approche A

# Etat NON CONFORME (fixe) — c'est la reference haute
LAMBDA_NON_CONF = 3.0
THETA_NON_CONF = 1.8


def _params_corps():
    mu = np.log(REM_MEDIANE)
    sigma = (np.log(REM_Q95) - mu) / stats.norm.ppf(0.95)
    return mu, sigma


def simuler_remediation(lam, rng):
    mu, sigma = _params_corps()
    n = rng.poisson(lam, size=M)
    tot = int(n.sum())
    en_q = rng.random(tot) < REM_P_QUEUE
    sev = np.empty(tot)
    nq = int(en_q.sum())
    if nq > 0:
        sev[en_q] = REM_SEUIL_POT + stats.genpareto.rvs(
            c=REM_XI, scale=REM_BETA, size=nq, random_state=rng)
    if tot - nq > 0:
        sev[~en_q] = rng.lognormal(mu, sigma, size=tot - nq)
    sev = np.minimum(sev, PLAFOND_INDIV)
    idx = np.cumsum(n)[:-1]
    return np.array([b.sum() if len(b) > 0 else 0.0 for b in np.split(sev, idx)])


def simuler_VaR(lam, theta, graine=GRAINE):
    """VaR 99,5% du L_DORA agrege pour un jeu (lambda, theta) donne."""
    rng = np.random.default_rng(graine)
    cal_s = calibrer_sanction(ca_annuel=800e6); L_s = simuler_sanction(cal_s, M, rng)
    L_r = simuler_remediation(lam, rng)
    cal_p = calibrer_prestataire(); L_p = simuler_prestataire(cal_p, M, rng)
    cal_a = calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2)
    L_a = simuler_aggravation(cal_a, M, rng)
    U = echantillon_gumbel(M, 4, theta, rng)
    marg = [L_s, L_r, L_p, L_a]
    L_dora = np.zeros(M)
    for j in range(4):
        L_dora += make_quantile_empirique(marg[j])(U[:, j])
    return np.quantile(L_dora, NIVEAU_VAR)


def main():
    profil = profil_assureur(ca_annuel=800e6)

    # VaR de l'etat NON CONFORME (commun a toutes les cellules)
    var_nc = simuler_VaR(LAMBDA_NON_CONF, THETA_NON_CONF)

    print("=" * 76)
    print(" SENSIBILITE DE Delta_DORA AUX HYPOTHESES D'EFFICACITE DE LA CONFORMITE")
    print("=" * 76)
    print(f"   Etat NON CONFORME (fixe) : lambda={LAMBDA_NON_CONF}, theta={THETA_NON_CONF}"
          f"  ->  VaR = {var_nc/1e6:.1f} M€")
    print(f"   Delta_DORA = VaR(non conforme) - VaR(conforme), pour divers etats conformes")
    print()

    # Grilles d'hypotheses pour l'etat CONFORME
    reductions_freq = [0.30, 0.50, 0.70]      # -30%, -50%, -70% de frequence
    thetas_conf = [1.4, 1.2, 1.1]             # dependance residuelle decroissante

    # En-tete
    print("   Delta_DORA (M€) selon (reduction frequence) x (theta conforme)")
    print("   " + "-" * 64)
    entete = "   red.freq \\ theta_c |" + "".join(f"  theta={t:<5.1f}" for t in thetas_conf)
    print(entete)
    print("   " + "-" * 64)

    grille = {}
    for rf in reductions_freq:
        lam_c = LAMBDA_NON_CONF * (1 - rf)
        ligne = f"   -{int(rf*100):>2d}% (lam={lam_c:.1f})    |"
        for tc in thetas_conf:
            var_c = simuler_VaR(lam_c, tc)
            delta = (var_nc - var_c) / 1e6
            grille[(rf, tc)] = delta
            ligne += f"  {delta:8.1f}"
        print(ligne)
    print("   " + "-" * 64)

    # Statistiques de la grille
    valeurs = np.array(list(grille.values()))
    centre = grille[(0.50, 1.2)]  # le scenario central d'approche_c
    print()
    print(f"   Scenario central (-50% freq, theta_c=1.2) : {centre:.1f} M€")
    print(f"   Fourchette complete de la grille          : "
          f"{valeurs.min():.1f} a {valeurs.max():.1f} M€")
    print(f"   Mediane de la grille                      : {np.median(valeurs):.1f} M€")

    # Decomposition : part frequence vs part dependance au scenario central
    print()
    print("   Decomposition au scenario central :")
    var_freq_only = simuler_VaR(LAMBDA_NON_CONF * 0.5, THETA_NON_CONF)  # theta inchange
    effet_freq = (var_nc - var_freq_only) / 1e6
    effet_theta = centre - effet_freq
    print(f"     - effet frequence seule (hygiene)     : {effet_freq:.1f} M€")
    print(f"     - effet dependance seule (sortie)     : {effet_theta:.1f} M€")

    print()
    print("=" * 76)
    print(" LECTURE POUR LA SOUTENANCE")
    print("=" * 76)
    print(f"""   - Delta_DORA n'est PAS un point unique : selon les hypotheses
     d'efficacite de la conformite, il varie de {valeurs.min():.0f} a {valeurs.max():.0f} M€.
   - Le scenario central (-50% frequence, theta 1.8->1.2) donne {centre:.0f} M€,
     mais ce chiffre est conditionnel : il faut le lire dans sa fourchette.
   - Plus on suppose la conformite efficace (forte reduction de frequence,
     faible dependance residuelle), plus le cout du non-respect est eleve.
   - Cette grille REMPLACE avantageusement un point unique : elle montre
     que la conclusion qualitative (la conformite libere des dizaines de
     M€ de capital) est ROBUSTE, meme si le montant exact ne l'est pas.
   => Honnetete methodologique : on ne pretend pas mesurer Delta_DORA, on
      borne son ordre de grandeur sous des hypotheses explicites. Les
      methodes causales (PSM, IPW, diff-in-diff) seraient requises pour
      l'estimer vraiment, mais exigent des donnees conformes/non-conformes
      reelles qui n'existent pas (contrefactuel non observable).""")
    print("=" * 76)


if __name__ == "__main__":
    main()
