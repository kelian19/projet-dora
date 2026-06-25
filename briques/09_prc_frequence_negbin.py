# 09_prc_frequence_negbin.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)

df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")
evt = df.drop_duplicates("group_uuid").copy()
evt = evt[(evt["reported_date"].dt.year >= 2021) & (evt["reported_date"].dt.year <= 2024)]

# Comptage MENSUEL
evt["mois"] = evt["reported_date"].dt.to_period("M")
m = evt.groupby("mois").size().reset_index(name="n")
m["t"] = np.arange(len(m))   # tendance lineaire
print(f"Nb de mois : {len(m)}")
print(f"Comptage mensuel : moyenne={m['n'].mean():.1f}, var={m['n'].var(ddof=1):.1f}")
print(f"Facteur brut mensuel (var/moy) : {m['n'].var(ddof=1)/m['n'].mean():.3f}")

# 1. Poisson avec tendance
poisson = smf.glm("n ~ t", data=m, family=sm.families.Poisson()).fit()
# 2. NegBin avec tendance (alpha = surdispersion)
negbin = smf.glm("n ~ t", data=m, family=sm.families.NegativeBinomial()).fit()

print("\n--- Test de surdispersion (Poisson) ---")
# Pearson chi2 / ddl : >1 = surdispersion
pearson = poisson.pearson_chi2 / poisson.df_resid
print(f"Pearson chi2 / ddl : {pearson:.3f}  (>1.2 => surdispersion)")

print("\n--- Modele NegBin ---")
print(f"AIC Poisson : {poisson.aic:.1f}  |  AIC NegBin : {negbin.aic:.1f}")
print(negbin.summary())