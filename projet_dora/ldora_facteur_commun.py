"""
=======================================================================
 SCR_DORA — Agregation par FACTEUR COMMUN + CHOC SYSTEMIQUE (Position 1)
=======================================================================
Alternative a la copule de Gumbel plate (echec d'agregation : marges
heterogenes -> echantillonneur Marshall-Olkin instable).

Principe :
  - chaque brique garde sa marge propre (fraicheur-severite, atome, scenario) ;
  - la dependance n'est PLUS portee par une copule sur les montants, mais par
    deux mecanismes economiques explicites :
      (a) un frailty doux  Z ~ Gamma(k, 1/k), E[Z]=1, qui module les
          intensites de frequence (surdispersion, regroupement des incidents) ;
      (b) un CHOC SYSTEMIQUE annuel partage B ~ Bernoulli(p_sys) qui, lorsqu'il
          se realise, declenche CONJOINTEMENT une remediation lourde, une
          sanction probable et un sinistre prestataire majeur (cf doc 5.2).

C'est (b) qui cree la dependance de QUEUE superieure. Elle ne se diagnostique
pas par le tau de Kendall global (pollue par la masse en zero des briques
rares), mais par la CONCENTRATION SYSTEMIQUE dans la queue de L_DORA :
      P(choc | L_DORA > VaR_99.5%)  >>  P(choc).

Resultat de reference : SCR_DORA ~ 85 M (meme ordre que l'approche A ~77 M),
borne par la somme comonotone des VaR marginales (cf garde-fou).
=======================================================================
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(20260616)

# ----------------------------------------------------------------------
# 1. Assureur synthetique (inchange vs approche A)
# ----------------------------------------------------------------------
CA, BSCR, FP = 800e6, 360e6, 784e6
CAP_SEV      = 50e6            # plafond de severite individuel (~10% FP)

# ----------------------------------------------------------------------
# 2. Severite remediation : corps lognormal + queue GPD (POT)
# ----------------------------------------------------------------------
MED, Q95 = 171e3, 9.3e6
MU_LN    = np.log(MED)
SIG_LN   = (np.log(Q95) - MU_LN) / stats.norm.ppf(0.95)
U_POT    = 5e6                 # seuil POT
XI_REF   = 1.4                 # forme (litterature), queue lourde xi>1
BETA_REF = U_POT * 0.9         # echelle GPD recalee sur marge remediation

def severite_remediation(n, xi, beta):
    """Melange corps lognormal (sous u) + u+GPD (queue), plafonne."""
    x = rng.lognormal(MU_LN, SIG_LN, size=n)
    tail = x > U_POT
    m = int(tail.sum())
    if m:
        u = rng.random(m)
        x[tail] = U_POT + beta / xi * ((1 - u) ** (-xi) - 1)
    return np.minimum(x, CAP_SEV)

# ----------------------------------------------------------------------
# 3. Parametres des briques et de la dependance
# ----------------------------------------------------------------------
# Frailty doux (surdispersion NegBin) — la queue vient du choc, pas de Z
K_FRAILTY = 2.5

# Choc systemique partage
P_SYS            = 0.03        # proba annuelle d'un choc systemique
LAMBDA_REM_SHOCK = 2.0         # surcroit d'intensite remediation si choc
P_SANCTION_SHOCK = 0.6         # proba de sanction conditionnelle au choc

# Frequences de base
LAMBDA_REM   = 3.0
P_SANCTION   = 0.04
LAMBDA_PREST = 0.10

# Sanction : c ~ Beta, montant = c * 2% CA
PLAFOND_SANCT = 0.02 * CA
A_BETA, B_BETA = 1.5, 5.0

# Prestataire : severite lognormale (scenarios experts agreges)
PREST_MU, PREST_SIG = np.log(18e6), 0.7

# ----------------------------------------------------------------------
# 4. Simulateur
# ----------------------------------------------------------------------
def simulate(M, k=K_FRAILTY, xi=XI_REF, beta=BETA_REF, cap=CAP_SEV,
             p_sys=P_SYS, return_B=False):
    global CAP_SEV
    CAP_SEV = cap

    Z = rng.gamma(k, 1.0 / k, size=M)        # frailty doux, E[Z]=1
    B = rng.random(M) < p_sys                 # choc systemique partage

    # --- Remediation ---
    n_rem = rng.poisson(LAMBDA_REM * Z + B * LAMBDA_REM_SHOCK)
    L_rem = np.zeros(M)
    idx = np.repeat(np.arange(M), n_rem)
    if idx.size:
        np.add.at(L_rem, idx, severite_remediation(idx.size, xi, beta))

    # --- Sanction (atome en zero) ---
    p_s = np.minimum(P_SANCTION * Z + B * P_SANCTION_SHOCK, 1.0)
    occ = rng.random(M) < p_s
    L_sanct = occ * rng.beta(A_BETA, B_BETA, size=M) * PLAFOND_SANCT

    # --- Prestataire (+1 sinistre majeur garanti si choc) ---
    n_pre = rng.poisson(LAMBDA_PREST * Z) + B.astype(int)
    L_pre = np.zeros(M)
    idxp = np.repeat(np.arange(M), n_pre)
    if idxp.size:
        sevp = np.minimum(rng.lognormal(PREST_MU, PREST_SIG, idxp.size), CAP_SEV)
        np.add.at(L_pre, idxp, sevp)

    if return_B:
        return L_rem, L_sanct, L_pre, B
    return L_rem, L_sanct, L_pre

VaR = lambda x, q=0.995: float(np.quantile(x, q))

# ----------------------------------------------------------------------
# 5. Run de reference
# ----------------------------------------------------------------------
if __name__ == "__main__":
    M = 1_000_000
    L_rem, L_sanct, L_pre, B = simulate(M, return_B=True)
    L_tot = L_rem + L_sanct + L_pre

    v_s, v_r, v_p = VaR(L_sanct), VaR(L_rem), VaR(L_pre)
    scr = VaR(L_tot)
    comonotone = v_s + v_r + v_p

    print("=== Marges stand-alone (VaR 99.5%) ===")
    print(f"  Sanction    : {v_s/1e6:6.1f} M")
    print(f"  Remediation : {v_r/1e6:6.1f} M")
    print(f"  Prestataire : {v_p/1e6:6.1f} M")
    print()
    print("=== Agregation par facteur commun + choc systemique ===")
    print(f"  SCR_DORA (VaR 99.5%)        : {scr/1e6:6.1f} M")
    print(f"  Borne comonotone (Sum VaR)  : {comonotone/1e6:6.1f} M  (garde-fou superieur)")
    print(f"  Benefice de diversification : {(1-scr/comonotone)*100:5.1f} %")

    print("\n=== Plausibilite reglementaire ===")
    print(f"  SCR / (0.30*BSCR=108M) : {scr/(0.30*BSCR):.2f}   (<1 attendu)")
    print(f"  SCR / CA               : {scr/CA:.3f}")
    print(f"  SCR / FP               : {scr/FP:.3f}")
    print(f"  SCR / formule std (32M): {scr/32e6:.2f}")

    print("\n=== GARDE-FOU dependance de queue : concentration systemique ===")
    tail = L_tot > scr
    print(f"  P(choc systemique)                 : {B.mean():.3f}")
    print(f"  P(choc | L_DORA > VaR 99.5%)       : {B[tail].mean():.3f}")
    print(f"  facteur de concentration           : x{B[tail].mean()/B.mean():.1f}")
    print("  -> la queue du SCR est gouvernee par le choc systemique partage")

    print("\n=== Sensibilite force de dependance k ===")
    for k in [1.0, 2.5, 5.0, 50.0]:
        Lr, Ls, Lp = simulate(M, k=k)
        print(f"  k={k:5.1f} (Var Z={1/k:5.3f})  SCR = {VaR(Lr+Ls+Lp)/1e6:6.1f} M")

    print("\n=== Sensibilite plafond de severite ===")
    for cap in [20e6, 50e6, 75e6, 100e6]:
        Lr, Ls, Lp = simulate(M, cap=cap)
        print(f"  plafond={cap/1e6:4.0f}M  SCR = {VaR(Lr+Ls+Lp)/1e6:6.1f} M")

    print("\n=== Sensibilite xi ===")
    for xi in [1.1, 1.25, 1.4, 1.5]:
        Lr, Ls, Lp = simulate(M, xi=xi, beta=U_POT * (xi / XI_REF) * 0.9)
        print(f"  xi={xi:.2f}  SCR = {VaR(Lr+Ls+Lp)/1e6:6.1f} M")

    print("\n=== Sensibilite proba de choc systemique p_sys ===")
    for p in [0.01, 0.03, 0.05, 0.10]:
        Lr, Ls, Lp = simulate(M, p_sys=p)
        print(f"  p_sys={p:.2f}  SCR = {VaR(Lr+Ls+Lp)/1e6:6.1f} M")
