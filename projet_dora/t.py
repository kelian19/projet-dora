import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculer_hill(data, k_max):
    data_tri = np.sort(data)
    n = len(data_tri)
    xi_hill, ic_bas, ic_haut = np.zeros(k_max), np.zeros(k_max), np.zeros(k_max)
    log_data = np.log(data_tri)
    
    for k in range(2, k_max + 1):
        log_exces = log_data[n-k : n]
        log_seuil = log_data[n-k-1]
        hill_val = np.mean(log_exces - log_seuil)
        xi_hill[k-1] = hill_val
        
        erreur_std = hill_val / np.sqrt(k)
        ic_bas[k-1] = hill_val - 1.96 * erreur_std
        ic_haut[k-1] = hill_val + 1.96 * erreur_std
        
    return xi_hill, ic_bas, ic_haut

def main():
    rng = np.random.default_rng(42)
    n_samples = 50_000
    
    # Paramètres issus de votre brique
    mediane_corps = 171_000.0
    q95_corps = 9_340_000.0
    seuil_pot = 4_950_000.0
    xi_vrai = 1.3
    beta = 2_000_000.0
    p_queue = 0.10
    
    mu = np.log(mediane_corps)
    sigma = (np.log(q95_corps) - mu) / stats.norm.ppf(0.95)
    
    en_q = rng.random(n_samples) < p_queue
    sev = np.empty(n_samples)
    nq = int(en_q.sum())
    
    if nq > 0:
        sev[en_q] = seuil_pot + stats.genpareto.rvs(c=xi_vrai, scale=beta, size=nq, random_state=rng)
    if n_samples - nq > 0:
        sev[~en_q] = rng.lognormal(mu, sigma, size=n_samples - nq)
        
    # Calcul et tracé
    k_max = 3000
    xi_hill, ic_bas, ic_haut = calculer_hill(sev, k_max)
    k_vals = np.arange(2, k_max + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_vals, xi_hill[1:], color="#1E3461", linewidth=2, label="Estimateur de Hill ($\hat{\\xi}_k$)")
    plt.fill_between(k_vals, ic_bas[1:], ic_haut[1:], color="#1E3461", alpha=0.15, label="IC à 95%")
    plt.axhline(y=xi_vrai, color="#2563EB", linestyle="--", linewidth=2, label=f"Vrai paramètre $\\xi = {xi_vrai}$")
    
    plt.title("Graphique de Hill (Validation de l'indice de queue cyber)", fontsize=14, fontweight="bold")
    plt.xlabel("Nombre de sinistres extrêmes retenus ($k$)", fontsize=12)
    plt.ylabel("Indice de queue estimé ($\\xi$)", fontsize=12)
    plt.ylim(0.5, 2.5)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("hill_plot_final.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()