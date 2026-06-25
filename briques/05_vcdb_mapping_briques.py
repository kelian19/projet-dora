# 05_vcdb_mapping_briques.py
import pandas as pd
import os

ROOT = r"C:\Users\KélianKADDOURI\projet-dora\briques"
df = pd.read_pickle(os.path.join(ROOT, "vcdb_df.pkl"))
N = len(df)

# 1. Attributs CIA (triade securite) — proxy structurel remediation/gravite
print("=== Attributs CIA (un incident peut cocher plusieurs) ===")
cia = {
    "Confidentiality (vol/divulgation donnees)": "attribute.Confidentiality",
    "Integrity (alteration)": "attribute.Integrity",
    "Availability (indisponibilite)": "attribute.Availability",
}
for label, col in cia.items():
    if col in df.columns:
        n = int(df[col].astype(bool).sum())
        print(f"  {label:<45} {n:>5}  ({n/N:.1%})")

# 2. Familles d'action (typologie technique)
print("\n=== Familles d'action ===")
for col in ["action.Hacking", "action.Malware", "action.Social",
            "action.Error", "action.Misuse", "action.Physical",
            "action.Environmental"]:
    if col in df.columns:
        n = int(df[col].astype(bool).sum())
        print(f"  {col.replace('action.',''):<20} {n:>5}  ({n/N:.1%})")

# 3. Proxy "availability" = remediation lourde, croise avec tiers
elargie_cols = [c for c in [
    "actor.Partner", "asset.ownership.Partner",
    "attribute.confidentiality.data_victim.Partner",
    "action.hacking.vector.Partner", "action.malware.vector.Partner",
    "action.physical.vector.Partner facility",
    "action.physical.vector.Partner vehicle",
] if c in df.columns]
tiers = df[elargie_cols].astype(bool).any(axis=1)

if "attribute.Availability" in df.columns:
    avail = df["attribute.Availability"].astype(bool)
    print("\n=== Croisement indisponibilite x tiers ===")
    print(f"  Indispo totale            : {int(avail.sum())}")
    print(f"  Indispo ET tiers          : {int((avail & tiers).sum())}")
    print(f"  Indispo SANS tiers        : {int((avail & ~tiers).sum())}")

# Export synthese mapping
mapping = pd.DataFrame([
    {"Brique": "Prestataire", "Mappage VERIS": "Direct (actor.Partner + signaux tiers)",
     "Exploitable": "Oui", "Usage": "Proportion / typologie"},
    {"Brique": "Remediation", "Mappage VERIS": "Proxy (attribute.Availability)",
     "Exploitable": "Partiel", "Usage": "Structure, pas calibration"},
    {"Brique": "Sanction", "Mappage VERIS": "Aucun (VERIS ne code pas le reglementaire)",
     "Exploitable": "Non", "Usage": "Dire d'expert"},
    {"Brique": "Aggravation", "Mappage VERIS": "Aucun (notion d'escalade non codee)",
     "Exploitable": "Non", "Usage": "Dire d'expert"},
])
mapping.to_csv(os.path.join(ROOT, "mapping_briques.csv"), index=False)
print("\n=== SYNTHESE MAPPING ===")
print(mapping.to_markdown(index=False))