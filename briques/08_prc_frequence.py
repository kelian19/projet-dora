# 08_prc_frequence.py
import pandas as pd
import numpy as np

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)

# Niveau evenement
df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")
df["annee"] = df["reported_date"].dt.year
evt = df.drop_duplicates("group_uuid").copy()

# >>> Fenetre homogene a choisir <
FENETRE = [2021, 2022, 2023, 2024]
comptes = evt[evt["annee"].isin(FENETRE)].groupby("annee").size()
print("Comptes evenementiels par annee :")
print(comptes)

lam = comptes.mean()
var = comptes.var(ddof=1)
facteur = var / lam
print(f"\nMoyenne (lambda)        : {lam:.1f}")
print(f"Variance empirique      : {var:.1f}")
print(f"Facteur surdispersion   : {facteur:.3f}")
print("-> NegBin justifiee" if facteur > 1.2 else "-> Proche Poisson")

# Filtre HACK seul (optionnel, pour comparaison)
evt_hack = evt[evt["breach_type"] == "HACK"]
comptes_h = evt_hack[evt_hack["annee"].isin(FENETRE)].groupby("annee").size()
print("\n--- HACK seul ---")
print(comptes_h)
if len(comptes_h) > 1:
    print(f"lambda HACK : {comptes_h.mean():.1f}  |  facteur : {comptes_h.var(ddof=1)/comptes_h.mean():.3f}")