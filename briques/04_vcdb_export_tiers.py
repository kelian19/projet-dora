# 04_vcdb_export_tiers.py
import pandas as pd
import os
import numpy as np

ROOT = r"C:\Users\KélianKADDOURI\projet-dora\briques"
df = pd.read_pickle(os.path.join(ROOT, "vcdb_df.pkl"))
N = len(df)

def prop_ic(n, total):
    p = n / total
    se = np.sqrt(p * (1 - p) / total)
    return p, max(0, p - 1.96*se), p + 1.96*se

# Définitions (identiques au script 03)
etroite = df["actor.Partner"].astype(bool)
signaux = [c for c in [
    "actor.Partner", "asset.ownership.Partner",
    "attribute.confidentiality.data_victim.Partner",
    "action.hacking.vector.Partner", "action.malware.vector.Partner",
    "action.social.vector.Partner",
    "action.physical.vector.Partner facility",
    "action.physical.vector.Partner vehicle",
] if c in df.columns]
elargie = df[signaux].astype(bool).any(axis=1)

# Table 1 : proportions
rows = []
for nom, m in [("Tiers acteur (etroite)", etroite),
               ("Tiers elargi (acteur/vecteur/actif)", elargie)]:
    n = int(m.sum())
    p, lo, hi = prop_ic(n, N)
    rows.append({"Definition": nom, "N_incidents": n, "N_total": N,
                 "Proportion": round(p, 4),
                 "IC95_bas": round(lo, 4), "IC95_haut": round(hi, 4)})
tab_prop = pd.DataFrame(rows)

# Table 2 : repartition par action (definition elargie)
acts = ["action.Hacking", "action.Malware", "action.Social",
        "action.Error", "action.Misuse", "action.Physical"]
rows2 = []
for a in acts:
    if a in df.columns:
        n = int((elargie & df[a].astype(bool)).sum())
        rows2.append({"Action": a.replace("action.", ""), "N_tiers_elargi": n})
tab_act = pd.DataFrame(rows2).sort_values("N_tiers_elargi", ascending=False)

# Exports
tab_prop.to_csv(os.path.join(ROOT, "tiers_proportions.csv"), index=False)
tab_act.to_csv(os.path.join(ROOT, "tiers_par_action.csv"), index=False)

print("=== PROPORTIONS ===")
print(tab_prop.to_markdown(index=False))
print("\n=== PAR ACTION (def. elargie) ===")
print(tab_act.to_markdown(index=False))
print(f"\nCSV exportes dans {ROOT}")