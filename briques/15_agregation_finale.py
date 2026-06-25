# -*- coding: utf-8 -*-
"""
=============================================================================
15. AGRÉGATION FINALE DORA — Copule de Gumbel & Allocation d'Euler
=============================================================================
Ce script orchestre l'ensemble du modèle interne cyber :
1. Génération des 4 distributions marginales (500 000 simulations).
2. Injection de la dépendance de queue (Copule de Gumbel).
3. Calcul du SCR DORA global (VaR 99.5%).
4. Allocation du capital par principe d'Euler.
"""

import numpy as np
import time

# --- Importation de tes modules (vérifie que ces noms correspondent à tes fichiers locaux) ---
import brique_remediation_corrigee as b_rem
import brique_prestataire as b_pres
import brique_sanction as b_sanc
import brique_aggravation as b_agg
import copule_gumbel_agregation as copule
import allocation_euler_var as euler

def main():
    M = 500_000
    rng = np.random.default_rng(42)
    THETA_GUMBEL = 1.5  # Paramètre de dépendance (1 = indép, >1 = queue lourde partagée)
    NIVEAU_VAR = 0.995

    print("=" * 70)
    print(" DÉMARRAGE DU MOTEUR D'AGRÉGATION DORA")
    print(f" Simulations : {M:,} | Copule Gumbel Theta : {THETA_GUMBEL}")
    print("=" * 70)

    t0 = time.time()

    # 1. Calibration des marginales
    # CORRECTION APPORTÉE ICI : calibrer_prestataire() au lieu de calibrer_scenarios()
    cal_rem = b_rem.calibrer_remediation()
    cal_pres = b_pres.calibrer_prestataire() 
    cal_sanc = b_sanc.calibrer_sanction()
    cal_agg = b_agg.calibrer_aggravation()

    # 2. Simulation des marginales (Stand-alone)
    print("\n>> Simulation des marginales en cours...")
    L_rem = b_rem.simuler_remediation(cal_rem, M, rng)
    L_pres = b_pres.simuler_prestataire(cal_pres, M, rng)
    L_sanc = b_sanc.simuler_sanction(cal_sanc, M, rng)
    L_agg = b_agg.simuler_aggravation(cal_agg, M, rng)

    marginales = [L_rem, L_pres, L_sanc, L_agg]
    noms_briques = ["Remédiation", "Prestataire", "Sanction", "Aggravation"]

    # Affichage Stand-alone
    print("\n>> RÉSULTATS STAND-ALONE (VaR 99.5%)")
    var_sa = []
    for nom, L in zip(noms_briques, marginales):
        var = np.quantile(L, NIVEAU_VAR)
        var_sa.append(var)
        print(f"   {nom:<15} : {var/1e6:>6.2f} M€")
    print("-" * 35)
    somme_sa = sum(var_sa)
    print(f"   SOMME DES VaR   : {somme_sa/1e6:>6.2f} M€  (Comonotonie)")

    # 3. Agrégation par Copule de Gumbel (Préservation des rangs)
    print("\n>> Application de la Copule de Gumbel (Dépendance de queue)...")
    
    # Génération de l'échantillon de copule et réordonnancement des marginales
    # On le fait explicitement ici pour pouvoir passer la matrice exacte à l'allocation d'Euler
    U = copule.echantillon_gumbel(M, 4, THETA_GUMBEL, rng)
    composantes_post_copule = []
    
    for j in range(4):
        marg_triee = np.sort(marginales[j])
        rangs = np.argsort(np.argsort(U[:, j]))
        composantes_post_copule.append(marg_triee[rangs])

    # Somme des composantes dépendantes pour obtenir la distribution globale
    L_dora_global = np.sum(composantes_post_copule, axis=0)
    scr_dora = np.quantile(L_dora_global, NIVEAU_VAR)
    
    benefice_div = somme_sa - scr_dora
    print(f"   SCR DORA GLOBAL : {scr_dora/1e6:>6.2f} M€")
    print(f"   Bénéfice de div.: {benefice_div/1e6:>6.2f} M€ ({-benefice_div/somme_sa:.1%})")

    # 4. Allocation d'Euler
    print("\n>> Allocation d'Euler (Répartition du capital diversifié)...")
    res_euler = euler.allocation_euler_var(composantes_post_copule, niveau=NIVEAU_VAR)
    
    for nom, alloc, sa in zip(noms_briques, res_euler["contributions"], var_sa):
        ratio = alloc / scr_dora
        print(f"   {nom:<15} : {alloc/1e6:>6.2f} M€  ({ratio:>5.1%}) | Stand-alone: {sa/1e6:>6.2f} M€")

    print(f"\n[Terminé en {time.time() - t0:.1f} secondes]")
    print("=" * 70)

if __name__ == "__main__":
    main()