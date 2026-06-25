# -*- coding: utf-8 -*-
"""
=============================================================================
VALIDATION OUT-OF-SAMPLE — INCIDENTS HACK POST-DORA
=============================================================================
Test du pouvoir prédictif du modèle sur les incidents HACK réels survenus 
APRÈS l'entrée en vigueur de DORA (Janvier 2025).
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

# Paramètres du modèle calibré en EUROS
MEDIANE_CORPS = 603_000.0
Q95_CORPS = 22_195_000.0
SEUIL_POT = 15_310_000.0
XI_EFF = 0.988
BETA = 21_000_000.0
P_QUEUE = 0.0654
PLAFOND_INDIV = 40_000_000.0

# Conversion Jacobs
JACOBS_A = 7.68
JACOBS_B = 0.76
USD_EUR = 0.92

DATE_DORA = pd.Timestamp("2025-01-17")
CSV_DEFAUT = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"

def charger_incidents_post_dora(chemin_csv):
    """Charge la base PRC (pipe-separated) et isole les HACK post-DORA."""
    df = pd.read_csv(chemin_csv, sep="|", encoding="utf-8-sig", low_memory=False)
    hack = df[df["breach_type"] == "HACK"].copy()

    def get_dt(row):
        for col in ["breach_date", "reported_date"]:
            v = row.get(col)
            if pd.notna(v):
                dt = pd.to_datetime(v, errors="coerce")
                if pd.notna(dt): return dt
        return pd.NaT

    hack["dt"] = hack.apply(get_dt, axis=1)
    post = hack[hack["dt"] >= DATE_DORA].copy()
    
    # On déduplique par group_uuid comme pour la calibration
    post = post.groupby("group_uuid").first().reset_index()
    
    records = pd.to_numeric(post["total_affected"], errors="coerce").dropna()
    return records[records > 1].values

def convertir_records_eur_jacobs(records):
    """Applique la formule de Jacobs pour convertir les records en euros."""
    cout_usd = np.exp(JACOBS_A + JACOBS_B * np.log(np.maximum(records, 1)))
    return cout_usd * USD_EUR

def cdf_mixte(x):
    """CDF de la loi mixte complète en Euros."""
    x = np.asarray(x, dtype=float)
    mu = np.log(MEDIANE_CORPS)
    sigma = (np.log(Q95_CORPS) - mu) / stats.norm.ppf(0.95)
    
    F_corps = stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    excedent = np.maximum(x - SEUIL_POT, 0.0)
    F_gpd = stats.genpareto.cdf(excedent, c=XI_EFF, scale=BETA)
    
    return (1.0 - P_QUEUE) * F_corps + P_QUEUE * F_gpd

def valider(couts):
    q_obs = cdf_mixte(couts)
    ks, pval = stats.kstest(q_obs, "uniform")
    n_depass = int((couts > PLAFOND_INDIV).sum())
    
    return {
        "n": len(couts), "cout_median": np.median(couts), "cout_max": couts.max(),
        "q_obs_median": np.median(q_obs), "ks_stat": ks, "ks_pval": pval,
        "n_depassement_plafond": n_depass
    }

def main():
    chemin = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAUT
    if not os.path.exists(chemin):
        print(f"[!] Fichier PRC introuvable : {chemin}")
        return
        
    records = charger_incidents_post_dora(chemin)
    if len(records) == 0:
        print("[!] Aucun incident HACK post-DORA trouvé dans le fichier.")
        return
        
    couts = convertir_records_eur_jacobs(records)
    res = valider(couts)
    
    print("=" * 70)
    print(" VALIDATION OUT-OF-SAMPLE : Incidents HACK post-DORA (>= Janvier 2025)")
    print("=" * 70)
    print(f"   Incidents exploitables : {res['n']}")
    print(f"   Coûts (Jacobs) observés : Médiane = {res['cout_median']/1e6:.2f} M€ | Max = {res['cout_max']/1e6:.2f} M€")
    print("-" * 70)
    
    verdict_ks = "COMPATIBLE (non rejeté)" if res["ks_pval"] > 0.05 else "ÉCART SIGNIFICATIF"
    print(f"   TEST 1 (Couverture) : Quantile médian = {res['q_obs_median']:.2f}")
    print(f"                         Test KS p-value = {res['ks_pval']:.3f} -> {verdict_ks}")
    print(f"   TEST 2 (Plafond)    : Dépassements du cap de {PLAFOND_INDIV/1e6:.0f} M€ = {res['n_depassement_plafond']}")
    print("=" * 70)

if __name__ == "__main__":
    main()