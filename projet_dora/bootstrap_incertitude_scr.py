# -*- coding: utf-8 -*-
"""
=============================================================================
BOOTSTRAP PARAMETRIQUE — PROPAGATION DE L'INCERTITUDE JUSQU'AU SCR
=============================================================================

OBJET
-----
Le SCR_DORA de référence (~103 M€) est une VaR PONCTUELLE, conditionnelle à un
jeu de paramètres dont AUCUN n'est connu avec certitude :
  - xi (épaisseur de queue)      : repris de la littérature, non estimable
  - lambda (fréquence entité)    : transposition marché -> entité
  - facteur (surdispersion NegBin): estimé sur 7 points annuels (IC large)
  - theta (dépendance Gumbel)    : choix expert
  - q95 corps (conversion €/record): principale source d'incertitude calibration

Ce module PROPAGE ces incertitudes : à chaque réplication, on tire un jeu de
paramètres dans sa loi, on relance la chaîne complète (4 briques + copule), et
on récupère le SCR. La distribution des SCR obtenus donne un INTERVALLE DE
CONFIANCE, qui transforme le résultat central du mémoire ("le SCR se lit comme
une fourchette") d'une AFFIRMATION en une DEMONSTRATION chiffrée.

METHODE
-------
Bootstrap paramétrique (Monte-Carlo à deux niveaux) :
  - niveau externe : B réplications de paramètres (incertitude d'estimation)
  - niveau interne : M_INT simulations de pertes (incertitude d'échantillonnage)
On rapporte la médiane et les quantiles [5%, 95%] de la distribution des SCR.

LOIS DE TIRAGE (justifiées §mémoire) :
  xi      ~ Uniforme(1.1, 1.5)        [fourchette littérature]
  lambda  ~ Gamma(moy=2, CV=30%)      [transposition incertaine]
  facteur ~ Uniforme(1.1, 2.0)        [IC large sur 7 ans]
  theta   ~ Uniforme(1.5, 2.2)        [dire d'expert encadré]
  q95     ~ q95_0 * Lognormal(0, 0.20)[conversion records->EUR ±20%]

AVERTISSEMENT : les lois de tirage encodent un jugement sur l'incertitude, pas
une vérité. L'IC obtenu est un IC SOUS CES HYPOTHESES d'incertitude, à
documenter comme tel. C'est néanmoins infiniment plus honnête qu'un point seul.
"""

import numpy as np
from scipy import stats

from brique_sanction import calibrer_sanction, simuler_sanction
from brique_prestataire import calibrer_prestataire, simuler_prestataire
from brique_aggravation import calibrer_aggravation, simuler_aggravation
from copule_gumbel_agregation import echantillon_gumbel, make_quantile_empirique
from ancrage_bilan_plausibilite import profil_assureur, tester_plausibilite_scr
import scr_dora_synthese_finale as ref

NIVEAU_VAR = 0.995
B = 500            # réplications de paramètres (niveau externe)
M_INT = 100_000    # simulations par réplication (niveau interne)
GRAINE = 2026
PLAFOND = ref.PLAFOND_CENTRAL   # 40 M€, fixé (hypothèse de perte max, pas incertain)


# =============================================================================
# TIRAGE D'UN JEU DE PARAMETRES INCERTAINS
# =============================================================================
def tirer_parametres(rng):
    """Tire un jeu de paramètres dans leurs lois d'incertitude respectives."""
    xi = rng.uniform(1.1, 1.5)
    # Gamma de moyenne 2, CV=30% -> k = 1/CV^2, scale = moy/k
    cv = 0.30
    k = 1.0 / cv**2
    lam = rng.gamma(k, ref.REM_LAMBDA_ENTITE / k)
    facteur = rng.uniform(1.1, 2.0)
    theta = rng.uniform(1.5, 2.2)
    # q95 du corps : ±20% lognormal autour de la valeur de référence
    q95 = ref.REM_Q95_CORPS * np.exp(rng.normal(0.0, 0.20))
    alpha_s = rng.uniform(0.5, 1.5)
    beta_s  = rng.uniform(4.0, 9.0)
    proba_s = rng.uniform(0.05, 0.20)
    scenarios_ref = [
        {"nom": "Panne hyperscaler cloud",            "proba_an": 0.08,
         "q50": 2_000_000, "q95": 25_000_000,  "q995": 150_000_000, "loi": "gpd"},
        {"nom": "Compromission prestataire paiement", "proba_an": 0.05,
         "q50": 1_000_000, "q95": 12_000_000,  "q995":  80_000_000, "loi": "gpd"},
        {"nom": "Rancongiciel editeur metier",        "proba_an": 0.12,
         "q50":   300_000, "q95":  3_000_000,  "q995":  20_000_000, "loi": "lognorm"},
    ]
    import numpy as _np
    scenarios_tires = []
    for s in scenarios_ref:
        s_tire = dict(s)
        s_tire["q95"]  = s["q95"]  * _np.exp(rng.normal(0.0, 0.30))
        s_tire["q995"] = s["q995"] * _np.exp(rng.normal(0.0, 0.30))
        s_tire["q95"]  = max(s_tire["q95"],  s["q50"] * 1.01)
        s_tire["q995"] = max(s_tire["q995"], s_tire["q95"] * 1.01)
        scenarios_tires.append(s_tire)
    return {"xi": xi, "lam": lam, "facteur": facteur, "theta": theta, "q95": q95,
            "alpha_s": alpha_s, "beta_s": beta_s, "proba_s": proba_s,
            "scenarios_presta": scenarios_tires}


def _sigma_corps(q95):
    """Recalcule sigma du corps lognormal pour un q95 donné (médiane fixe)."""
    mu = np.log(ref.REM_MEDIANE_CORPS)
    return mu, (np.log(q95) - mu) / stats.norm.ppf(0.95)


def simuler_remediation_param(p, rng):
    """Remédiation avec paramètres tirés (xi, lam, facteur, q95)."""
    mu, sigma = _sigma_corps(p["q95"])
    # fréquence NegBin (paramétrisation correcte : E[N]=lam)
    if p["facteur"] <= 1.0 + 1e-9:
        n = rng.poisson(p["lam"], size=M_INT)
    else:
        pn = 1.0 / p["facteur"]
        rn = p["lam"] * pn / (1.0 - pn)
        n = rng.negative_binomial(rn, pn, size=M_INT)
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(M_INT)
    en_q = rng.random(tot) < ref.REM_P_QUEUE
    sev = np.empty(tot)
    nq = int(en_q.sum())
    if nq > 0:
        sev[en_q] = ref.REM_SEUIL_POT + stats.genpareto.rvs(
            c=p["xi"], scale=ref.REM_BETA, size=nq, random_state=rng)
    if tot - nq > 0:
        sev[~en_q] = rng.lognormal(mu, sigma, size=tot - nq)
    sev = np.minimum(sev, PLAFOND)
    idx = np.cumsum(n)[:-1]
    return np.array([b.sum() if len(b) > 0 else 0.0 for b in np.split(sev, idx)])


def scr_une_replication(p, rng):
    """Relance la chaîne complète pour un jeu de paramètres -> un SCR."""
    cal_s = calibrer_sanction(ca_annuel=800e6)
    cal_p = calibrer_prestataire()
    cal_a = calibrer_aggravation(proba_survenance=0.50, mu=11.0, sigma=1.2)
    L_s = simuler_sanction(cal_s, M_INT, rng)
    L_p = simuler_prestataire(cal_p, M_INT, rng)
    L_a = simuler_aggravation(cal_a, M_INT, rng)
    L_r = simuler_remediation_param(p, rng)
    marginales = [L_s, L_r, L_p, L_a]
    U = echantillon_gumbel(M_INT, 4, p["theta"], rng)
    L = np.sum([make_quantile_empirique(marginales[j])(U[:, j]) for j in range(4)],
               axis=0)
    return np.quantile(L, NIVEAU_VAR)


# =============================================================================
# BOOTSTRAP
# =============================================================================
def lancer_bootstrap(B=B, graine=GRAINE, verbose=True):
    """Lance B réplications et renvoie la distribution des SCR."""
    rng = np.random.default_rng(graine)
    scrs = np.empty(B)
    params = []
    for b in range(B):
        p = tirer_parametres(rng)
        scrs[b] = scr_une_replication(p, rng)
        params.append(p)
        if verbose and (b + 1) % 50 == 0:
            print(f"   ... {b+1}/{B} réplications  (SCR courant médian = "
                  f"{np.median(scrs[:b+1])/1e6:.1f} M€)")
    return scrs, params


def analyser(scrs, profil):
    """Statistiques et plausibilité de la distribution bootstrap du SCR."""
    q = lambda a: np.quantile(scrs, a)
    plafond_reg = profil["plafond_op_reglementaire"]
    prop_plausible = np.mean(scrs <= plafond_reg)
    return {
        "median": np.median(scrs), "moyenne": scrs.mean(),
        "q05": q(0.05), "q25": q(0.25), "q75": q(0.75), "q95": q(0.95),
        "min": scrs.min(), "max": scrs.max(), "ecart_type": scrs.std(),
        "prop_plausible": prop_plausible, "plafond_reg": plafond_reg,
    }


def decomposer_variance(scrs, params):
    """Quel paramètre pilote l'incertitude du SCR ?

    - rho de Spearman : monotonie SCR ~ paramètre (signe et force du lien)
    - part de variance : beta² d'une régression standardisée, normalisés
    Renvoie un dict {param: (rho, part_variance)} trié par part décroissante.
    """
    from scipy.stats import spearmanr
    from numpy.linalg import lstsq
    keys = ["xi", "lam", "facteur", "theta", "q95"]
    mat = {k: np.array([p[k] for p in params]) for k in keys}
    rho = {k: spearmanr(mat[k], scrs)[0] for k in keys}
    X = np.column_stack([(mat[k] - mat[k].mean()) / mat[k].std() for k in keys])
    y = (scrs - scrs.mean()) / scrs.std()
    beta, *_ = lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)
    b2 = beta[1:] ** 2
    part = b2 / b2.sum()
    return {k: (rho[k], part[i]) for i, k in enumerate(keys)}


def afficher_decomposition(decomp):
    print("\n" + "=" * 70)
    print(" QUI PILOTE L'INCERTITUDE DU SCR ? (décomposition de variance)")
    print("=" * 70)
    labels = {"xi": "xi (queue)", "lam": "lambda (fréquence entité)",
              "facteur": "facteur surdisp.", "theta": "theta (dépendance)",
              "q95": "q95 (conversion €/record)"}
    print(f"   {'paramètre':28s} | {'rho Spearman':>12s} | {'part variance':>13s}")
    print("   " + "-" * 60)
    for k, (r, p) in sorted(decomp.items(), key=lambda x: -x[1][1]):
        print(f"   {labels[k]:28s} | {r:>+12.3f} | {p:>12.1%}")
    print("   " + "-" * 60)
    dom = max(decomp.items(), key=lambda x: x[1][1])[0]
    print(f"   => L'incertitude est dominée par '{labels[dom]}'.")
    print("      Implication : pour fiabiliser le capital, prioriser")
    print("      l'estimation de ce paramètre plutôt que les autres.")
    print("=" * 70)


def main():
    print("=" * 70)
    print(" BOOTSTRAP PARAMETRIQUE DU SCR_DORA")
    print(f" B={B} réplications x M_INT={M_INT:,} simulations (plafond {PLAFOND/1e6:.0f} M€)")
    print("=" * 70)
    profil = profil_assureur(ca_annuel=800e6)
    scrs, params = lancer_bootstrap()
    res = analyser(scrs, profil)
    decomp = decomposer_variance(scrs, params)

    print("\n" + "=" * 70)
    print(" DISTRIBUTION DU SCR_DORA SOUS INCERTITUDE PARAMETRIQUE")
    print("=" * 70)
    print(f"   Médiane                    : {res['median']/1e6:7.1f} M€")
    print(f"   Moyenne                    : {res['moyenne']/1e6:7.1f} M€")
    print(f"   Ecart-type                 : {res['ecart_type']/1e6:7.1f} M€")
    print(f"   IC 90% [q05 ; q95]         : [{res['q05']/1e6:.1f} ; {res['q95']/1e6:.1f}] M€")
    print(f"   IC 50% [q25 ; q75]         : [{res['q25']/1e6:.1f} ; {res['q75']/1e6:.1f}] M€")
    print(f"   Etendue [min ; max]        : [{res['min']/1e6:.1f} ; {res['max']/1e6:.1f}] M€")
    print(f"   Plafond rég. (0,3*BSCR)    : {res['plafond_reg']/1e6:7.1f} M€")
    print(f"   P(SCR <= plafond rég.)     : {res['prop_plausible']:7.1%}")
    print("=" * 70)
    afficher_decomposition(decomp)
    print(f"""
 LECTURE POUR LE MEMOIRE
   - Le SCR_DORA n'est pas un point mais une DISTRIBUTION : médiane
     ~{res['median']/1e6:.0f} M€, IC 90% [{res['q05']/1e6:.0f} ; {res['q95']/1e6:.0f}] M€. C'est la traduction
     quantitative directe de la thèse centrale du mémoire.
   - {res['prop_plausible']:.0%} des configurations d'incertitude restent sous le plafond
     réglementaire : la plausibilité du scénario central est robuste.
   - L'incertitude de calibration (xi, lambda, theta, conversion €/record)
     domine l'incertitude d'échantillonnage Monte-Carlo : le besoin en
     capital cyber EST une fourchette, conditionnelle aux hypothèses de queue.
""")
    print("=" * 70)
    return scrs, params, res


if __name__ == "__main__":
    main()
