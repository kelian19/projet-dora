# -*- coding: utf-8 -*-
"""
=============================================================================
APPROCHE C — CONTREFACTUEL DORA (conforme vs non-conforme)
=============================================================================

Mesure la perte imputable au NON-RESPECT de DORA comme l'écart de capital
entre deux états du monde simulés sur un PROFIL DE RISQUE IDENTIQUE :

    Delta_DORA = VaR_99,5%(L | non conforme) - VaR_99,5%(L | conforme)

La conformité n'est pas une perte directe : elle agit sur les PARAMETRES.
  - fréquence : lambda_conf < lambda_non_conf
        (hygiène cyber -> moins d'incidents réussis)
  - dépendance de queue : theta_conf < theta_non_conf
        (stratégies de sortie, redondance -> moins de contagion entre briques)

NEUTRALISATION DU BIAIS DE SELECTION (cf. doc §3.5)
--------------------------------------------------
On ne compare PAS deux populations réelles (conformes vs non-conformes du
marché), ce qui mélangerait l'effet DORA et une différence de profil
préexistante. On simule UN SEUL profil synthétique, deux fois, avec la MEME
graine et les MEMES marginales de sévérité ; seuls lambda et theta changent.
Le profil sous-jacent étant identique, l'écart ne peut pas être contaminé.

STATUT DU RESULTAT
------------------
Delta_DORA est une BORNE SUPERIEURE de l'effet propre du non-respect. Les
rapports lambda_conf/lambda_non_conf et theta_conf/theta_non_conf sont calibrés
A DIRE D'EXPERT / revue de littérature sur l'efficacité des contrôles, pas
estimés. Le sens du biais résiduel (surestimation) est documenté.

AVERTISSEMENT : contrefactuel non observable. Résultat conditionnel aux
hypothèses d'efficacité de la conformité. A présenter en OUVERTURE CRITIQUE,
l'approche A (taxonomie) restant le socle du mémoire.
"""

import numpy as np
from scipy import stats

from brique_sanction import calibrer_sanction, simuler_sanction
from brique_prestataire import calibrer_prestataire, simuler_prestataire
from brique_aggravation import calibrer_aggravation, simuler_aggravation
from copule_gumbel_agregation import echantillon_gumbel, make_quantile_empirique
from ancrage_bilan_plausibilite import profil_assureur, afficher_plausibilite

NIVEAU_VAR = 0.995
M = 500_000
GRAINE = 42

# Paramètres remédiation (issus PRC HACK réels, cf. synthèse finale)
REM_MEDIANE_CORPS = 171_000.0
REM_Q95_CORPS = 9_340_000.0
REM_SEUIL_POT = 4_950_000.0
REM_BETA = 2_000_000.0
REM_P_QUEUE = 0.10
REM_XI = 1.3
REM_FACTEUR_SURDISP = 1.30     # variance/moyenne (NegBin) calibré sur PRC HACK
                               # (alpha~0,15 marché transposé à lambda entité) ; 1 -> Poisson
PLAFOND_INDIV = 40_000_000.0   # scénario central retenu (cohérent approche A)

# --- HYPOTHESES D'EFFICACITE DE LA CONFORMITE (dire d'expert) ---
# Fréquence : la conformité réduit le nombre d'incidents réussis.
LAMBDA_NON_CONF = 3.0          # entité non conforme : plus d'incidents
RATIO_LAMBDA_CONF = 0.5        # conformité -> -50% de fréquence
# Dépendance de queue (copule Gumbel) entre les 4 briques.
THETA_NON_CONF = 1.8           # forte contagion (concentration cloud non maîtrisée)
THETA_CONF = 1.2               # faible contagion (stratégies de sortie actives)


def _params_corps():
    mu = np.log(REM_MEDIANE_CORPS)
    sigma = (np.log(REM_Q95_CORPS) - mu) / stats.norm.ppf(0.95)
    return mu, sigma


def simuler_remediation(lam, rng):
    """Remédiation corps+queue, fréquence binomiale négative (surdispersion), plafond central."""
    mu, sigma = _params_corps()
    if REM_FACTEUR_SURDISP <= 1.0 + 1e-9:
        n = rng.poisson(lam, size=M)
    else:
        var = REM_FACTEUR_SURDISP * lam
        p = lam / var
        r = lam * p / (1.0 - p)
        n = rng.negative_binomial(r, p, size=M)
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


def simuler_monde(conforme, graine, theta_conf=THETA_CONF):
    """
    Simule L_DORA pour un état du monde (conforme ou non).
    MEME graine des deux côtés -> profil sous-jacent identique.
    Seuls lambda (fréquence rem.) et theta (copule) diffèrent.
    theta_conf est paramétrable pour la décomposition de l'effet.
    """
    rng = np.random.default_rng(graine)

    lam = LAMBDA_NON_CONF * (RATIO_LAMBDA_CONF if conforme else 1.0)
    theta = theta_conf if conforme else THETA_NON_CONF

    # Marginales (mêmes lois des deux côtés ; seule la fréquence rem. change)
    cal_s = calibrer_sanction(ca_annuel=800e6)
    cal_p = calibrer_prestataire()
    cal_a = calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2)

    L_s = simuler_sanction(cal_s, M, rng)
    L_r = simuler_remediation(lam, rng)
    L_p = simuler_prestataire(cal_p, M, rng)
    L_a = simuler_aggravation(cal_a, M, rng)
    marginales = [L_s, L_r, L_p, L_a]

    # Agrégation par copule de Gumbel (dépendance theta)
    U = echantillon_gumbel(M, 4, theta, rng)
    composantes = []
    for j in range(4):
        q = make_quantile_empirique(marginales[j])
        composantes.append(q(U[:, j]))
    L_dora = np.sum(composantes, axis=0)

    return {
        "lambda": lam, "theta": theta,
        "L_dora": L_dora,
        "VaR": np.quantile(L_dora, NIVEAU_VAR),
        "moyenne": L_dora.mean(),
    }


def main():
    profil = profil_assureur(ca_annuel=800e6)

    print("=" * 74)
    print(" APPROCHE C — CONTREFACTUEL DORA (profil constant)")
    print("=" * 74)
    print(f"   Hypothèses d'efficacité de la conformité (dire d'expert) :")
    print(f"     fréquence : lambda {LAMBDA_NON_CONF:.1f} -> "
          f"{LAMBDA_NON_CONF*RATIO_LAMBDA_CONF:.1f} (-{(1-RATIO_LAMBDA_CONF)*100:.0f}%)")
    print(f"     dépendance: theta  {THETA_NON_CONF:.1f} -> {THETA_CONF:.1f}")
    print(f"     (tau Kendall {1-1/THETA_NON_CONF:.2f} -> {1-1/THETA_CONF:.2f})")

    # MEME graine des deux côtés : profil sous-jacent identique
    monde_nc = simuler_monde(conforme=False, graine=GRAINE)
    monde_c = simuler_monde(conforme=True, graine=GRAINE)

    delta = monde_nc["VaR"] - monde_c["VaR"]

    print("\n" + "=" * 74)
    print(" RESULTATS")
    print("=" * 74)
    print(f"   {'Etat':16s} | {'lambda':>7s} | {'theta':>6s} | {'VaR 99,5%':>12s}")
    print("   " + "-" * 52)
    print(f"   {'Non conforme':16s} | {monde_nc['lambda']:7.1f} | "
          f"{monde_nc['theta']:6.1f} | {monde_nc['VaR']/1e6:9.1f} M€")
    print(f"   {'Conforme':16s} | {monde_c['lambda']:7.1f} | "
          f"{monde_c['theta']:6.1f} | {monde_c['VaR']/1e6:9.1f} M€")
    print("   " + "-" * 52)
    print(f"   Delta_DORA (coût du non-respect, BORNE SUP) : {delta/1e6:7.1f} M€")
    print(f"   Réduction relative de capital               : "
          f"{100*delta/monde_nc['VaR']:6.1f} %")

    # Décomposition de l'effet : fréquence seule vs dépendance seule
    print("\n" + "=" * 74)
    print(" DECOMPOSITION DE L'EFFET (quelle part vient de quoi ?)")
    print("=" * 74)
    # Effet fréquence seule (theta du conforme fixé au niveau non-conforme)
    monde_c_freqonly = simuler_monde(conforme=True, graine=GRAINE,
                                     theta_conf=THETA_NON_CONF)
    effet_freq = monde_nc["VaR"] - monde_c_freqonly["VaR"]
    effet_theta = delta - effet_freq
    print(f"   Effet fréquence (hygiène cyber)   : {effet_freq/1e6:6.1f} M€ "
          f"({100*effet_freq/delta:4.0f}% du total)")
    print(f"   Effet dépendance (stratégie sortie): {effet_theta/1e6:6.1f} M€ "
          f"({100*effet_theta/delta:4.0f}% du total)")

    print("\n" + "=" * 74)
    print(" PLAUSIBILITE DES DEUX ETATS")
    print("=" * 74)
    for nom, monde in [("NON CONFORME", monde_nc), ("CONFORME", monde_c)]:
        from ancrage_bilan_plausibilite import tester_plausibilite_scr
        t = tester_plausibilite_scr(monde["VaR"], profil)
        print(f"   {nom:13s}: SCR={monde['VaR']/1e6:6.1f}M€ -> "
              f"{'PLAUSIBLE' if t['plausible'] else 'NON PLAUSIBLE'}")

    print("\n" + "=" * 74)
    print(" LECTURE POUR LA SOUTENANCE")
    print("=" * 74)
    print(f"""   - Delta_DORA = {delta/1e6:.0f} M€ est le coût en capital du non-respect,
     calculé à profil constant (biais de sélection neutralisé).
   - C'est une BORNE SUPERIEURE : les ratios d'efficacité (lambda, theta)
     sont des hypothèses expert, non estimées. Le sens du biais résiduel
     est la surestimation.
   - A présenter en OUVERTURE CRITIQUE : le contrefactuel est non
     observable. Le socle du mémoire reste l'approche A (périmètre par
     taxonomie), objective et reproductible.
   - L'écart se lit comme un argument métier : investir dans la conformité
     (hygiène + plans de sortie) libère ~{delta/1e6:.0f} M€ de capital
     réglementaire — un retour sur fonds propres, pas un centre de coût.""")
    print("=" * 74)


if __name__ == "__main__":
    main()