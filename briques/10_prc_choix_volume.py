# 10_prc_choix_volume.py
import pandas as pd
import numpy as np

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)

# Niveau EVENEMENT (dedup group_uuid). Pour total_affected, le volume
# de l'evenement est le meme sur toutes les notifs -> on prend le max par groupe.
# Pour residents_affected, c'est un compte PAR ETAT -> on teste les deux logiques.
df["total_affected"] = pd.to_numeric(df["total_affected"], errors="coerce")
df["residents_affected"] = pd.to_numeric(df["residents_affected"], errors="coerce")

# Agregation au niveau evenement
g = df.groupby("group_uuid")
evt = pd.DataFrame({
    "total_affected": g["total_affected"].max(),          # volume global de l'evenement
    "residents_sum": g["residents_affected"].sum(min_count=1),  # somme inter-Etats
    "residents_max": g["residents_affected"].max(),       # max inter-Etats
})

N_evt = len(evt)
print(f"Evenements uniques : {N_evt:,}\n")

def profil(nom, s):
    s = s.dropna()
    s = s[s > 1]   # on exclut blank/0/1 comme le fait PRC officiellement
    if len(s) == 0:
        print(f"--- {nom} : VIDE ---"); return
    ql = s.quantile([.5, .9, .95, .99, .999])
    print(f"--- {nom} ---")
    print(f"  n (>1)        : {len(s):,}  ({len(s)/N_evt:.1%} des evenements)")
    print(f"  moyenne       : {s.mean():,.0f}")
    print(f"  mediane (Q50) : {ql[.5]:,.0f}")
    print(f"  Q90 / Q95     : {ql[.9]:,.0f} / {ql[.95]:,.0f}")
    print(f"  Q99 / Q99.9   : {ql[.99]:,.0f} / {ql[.999]:,.0f}")
    print(f"  max           : {s.max():,.0f}")
    # Indice de queue lourde : ratio moyenne/mediane
    print(f"  moyenne/mediane : {s.mean()/ql[.5]:,.1f}  (>10 => queue tres lourde)\n")

profil("total_affected (max par evt)", evt["total_affected"])
profil("residents_affected (somme inter-Etats)", evt["residents_sum"])
profil("residents_affected (max inter-Etats)", evt["residents_max"])

# Recouvrement : combien d'evenements ont les DEUX renseignes
both = evt.dropna(subset=["total_affected", "residents_sum"])
both = both[(both["total_affected"] > 1) & (both["residents_sum"] > 1)]
print(f"=== Recouvrement ===")
print(f"Evenements avec total ET residents (>1) : {len(both):,}")
if len(both) > 0:
    ratio = (both["total_affected"] / both["residents_sum"]).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Ratio total/residents : mediane={ratio.median():.1f}, "
          f"Q25={ratio.quantile(.25):.1f}, Q75={ratio.quantile(.75):.1f}")