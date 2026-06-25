# 07_prc_diagnostic.py
import pandas as pd

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)

print("Lignes (notifications) :", len(df))
print("Evenements uniques (group_uuid) :", df["group_uuid"].nunique())
print("Ratio notifications/evenement :", round(len(df) / df["group_uuid"].nunique(), 2))

for col in ["total_affected", "residents_affected"]:
    s = pd.to_numeric(df[col], errors="coerce")
    print(f"\n--- {col} ---")
    print(f"  non-nuls : {s.notna().sum():,}  ({s.notna().mean():.1%})")
    print(f"  >1       : {(s > 1).sum():,}")
    print(f"  mediane / max : {s.median():.0f} / {s.max():.0f}")

print("\n--- breach_type (top) ---")
print(df["breach_type"].value_counts().head(10))

df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")
df["annee"] = df["reported_date"].dt.year
evt = df.drop_duplicates("group_uuid")
print("\n--- Evenements uniques par annee (reported_date) ---")
print(evt["annee"].value_counts().sort_index())