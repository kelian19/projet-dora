# 09c_prc_negbin_alpha.py
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

y = m["n"].astype(float)
X = np.ones((len(y), 1))   # intercept seul (tendance non significative)

# Valeurs de depart raisonnables : log(moyenne) pour l'intercept, alpha modere
start = [np.log(y.mean()), 1.0]
mod = NegativeBinomialP(y, X, p=2).fit(start_params=start, method="nm",
                                       maxiter=5000, disp=False)
mod = NegativeBinomialP(y, X, p=2).fit(start_params=mod.params, disp=False)  # raffinage
print(mod.summary())

alpha = mod.params["alpha"]
mu = y.mean()
print(f"\nconverged : {mod.mle_retvals['converged']}")
print(f"alpha (MLE)            : {alpha:.5f}")
print(f"mu mensuel moyen       : {mu:.1f}")
print(f"Facteur surdispersion implique (1 + alpha*mu) : {1 + alpha*mu:.2f}")