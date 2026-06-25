# 02_vcdb_explore_partner.py
# Repérage des colonnes liées au tiers/partenaire/supply chain
import pandas as pd
import os

# Le pickle est dans le dossier briques, à côté des scripts
ROOT = r"C:\Users\KélianKADDOURI\projet-dora\briques"
PKL = os.path.join(ROOT, "vcdb_df.pkl")

df = pd.read_pickle(PKL)

# 1. Colonnes "actor.partner" (acteur = un partenaire/tiers)
partner_cols = [c for c in df.columns if "actor.partner" in c.lower()]
print("=== actor.partner ===")
for c in partner_cols:
    print(f"  {c:<60} {pd.to_numeric(df[c], errors='coerce').sum():>6.0f}")

# 2. Colonnes mentionnant supply chain / third party
supply_cols = [c for c in df.columns
               if any(k in c.lower() for k in ["supply", "third", "partner"])]
print("\n=== supply / third / partner (toutes) ===")
for c in supply_cols:
    print(f"  {c:<60} {pd.to_numeric(df[c], errors='coerce').sum():>6.0f}")

# 3. Vue d'ensemble des grandes familles d'acteurs (pour le mapping briques)
print("\n=== Familles d'acteurs (actor.*) ===")
for fam in ["actor.external", "actor.internal", "actor.partner"]:
    col = fam  # verispy crée souvent une colonne agrégée
    matches = [c for c in df.columns if c.lower() == fam.lower()]
    if matches:
        print(f"  {fam:<25} {df[matches[0]].sum():>6.0f}")
    else:
        # sinon on somme les sous-colonnes
        subs = [c for c in df.columns if c.lower().startswith(fam.lower()+".")]
        print(f"  {fam:<25} (pas de colonne agrégée, {len(subs)} sous-colonnes)")