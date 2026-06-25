# 09d_prc_negbin_robuste.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomialP

PATH = r"C:\Users\KélianKADDOURI\projet-dora\briques\Data_Breach_Chronology.csv"
df = pd.read_csv(PATH, sep="|", encoding="utf-8-sig", low_memory=False)
df["reported_date"] = pd.to_datetime(df["reported_date"], errors="coerce")
evt = df.drop_duplicates("group_uuid").copy()
evt = evt[(evt["reported_date"].dt.year >= 2021) & (evt["reported_date"].dt.year <= 2024)]
evt["mois"] = evt["reported_date"].dt.to_period("M")
m = evt.groupby("mois").size().reset_index(name="n")

print("=== Profil mensuel ===")
print(m["n"].describe())
print("\n=== Top 6 mois (pics potentiels) ===")
print(m.sort_values("n", ascending=False).head(6).to_string(index=False))

def estime_alpha(y):
    X = np.ones((len(y), 1))
    start = [np.log(y.mean()), 1.0]
    mod = NegativeBinomialP(y, X, p=2).fit(start_params=start, method="nm",
                                           maxiter=5000, disp=False)
    a = mod.params["alpha"]
    return a, 1 + a*y.mean()

# alpha complet
a_full, f_full = estime_alpha(m["n"].astype(float))

# alpha en retirant les mois au-dela de mediane + 3*IQR (outliers hauts)
q1, q3 = m["n"].quantile([.25, .75])
seuil = q3 + 1.5*(q3 - q1)
m_clean = m[m["n"] <= seuil]
a_clean, f_clean = estime_alpha(m_clean["n"].astype(float))

print(f"\n=== Sensibilite ===")
print(f"Complet    : {len(m)} mois, alpha={a_full:.4f}, facteur={f_full:.2f}")
print(f"Sans pics  : {len(m_clean)} mois (seuil={seuil:.0f}), alpha={a_clean:.4f}, facteur={f_clean:.2f}")
print(f"Mois retires : {len(m) - len(m_clean)}")