# 03_vcdb_tiers_brique.py
import pandas as pd
import os
import numpy as np

ROOT = r"C:\Users\KélianKADDOURI\projet-dora\briques"
df = pd.read_pickle(os.path.join(ROOT, "vcdb_df.pkl"))
N = len(df)

def prop_ic(n_succes, n_total):
    """Proportion + IC95% Wald."""
    p = n_succes / n_total
    se = np.sqrt(p * (1 - p) / n_total)
    return p, p - 1.96*se, p + 1.96*se

# --- Définition ÉTROITE : le tiers est l'acteur ---
etroite = df["actor.Partner"].astype(bool)

# --- Définition ÉLARGIE : tiers acteur OU vecteur OU actif/donnée tiers ---
signaux = [
    "actor.Partner",
    "asset.ownership.Partner",
    "attribute.confidentiality.data_victim.Partner",
    "action.hacking.vector.Partner",
    "action.malware.vector.Partner",
    "action.social.vector.Partner",
    "action.physical.vector.Partner facility",
    "action.physical.vector.Partner vehicle",
]
signaux = [c for c in signaux if c in df.columns]  # garde-fou
elargie = df[signaux].astype(bool).any(axis=1)

for nom, masque in [("ÉTROITE (acteur tiers)", etroite),
                    ("ÉLARGIE (acteur/vecteur/actif tiers)", elargie)]:
    n = int(masque.sum())
    p, lo, hi = prop_ic(n, N)
    print(f"{nom:<42} n={n:>4}  p={p:6.2%}  IC95=[{lo:.2%}, {hi:.2%}]")

# Croisement avec le type d'action dominant (pour le mapping briques)
print("\n--- Répartition des incidents 'tiers élargi' par action ---")
for act in ["action.Hacking", "action.Malware", "action.Social",
            "action.Error", "action.Misuse", "action.Physical"]:
    if act in df.columns:
        n = int((elargie & df[act].astype(bool)).sum())
        print(f"  {act:<20} {n:>4}")