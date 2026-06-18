# -*- coding: utf-8 -*-
"""
=============================================================================
DIAGNOSTIC EVT — GRAPHIQUE DE HILL (HILL PLOT)
=============================================================================
Génère le Hill Plot pour justifier le paramètre d'épaisseur de queue (xi).
Affiche l'estimateur de Hill en fonction du nombre de statistiques d'ordre (k).
La zone de stabilité ("plateau") indique le vrai paramètre de queue.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Paramètres de votre modèle (pour générer l'échantillon de test)
MEDIANE_CORPS = 171_000.0
Q95_CORPS = 9_340_000.0
SEUIL_POT = 4_950_000.0
XI_VRAI = 1.3
BETA = 2_000_000.0
P_QUEUE = 0.10

def generer_donnees_test(n_samples=20000, rng=None):
    """Génère des pseudo-données avec le raccord Lognormale / GPD"""
    mu = np.log(MEDIANE_CORPS)
    sigma = (np.log(Q95_CORPS) - mu) / stats.norm.ppf(0.95)
    en_q = rng.random(n_samples) < P_QUEUE
    sev = np.empty(n_samples)
    nq = int(en_q.sum())
    
    if nq > 0:
        sev[en_q] = SEUIL_POT + stats.genpareto.rvs(c=XI_VRAI, scale=BETA, size=nq, random_state=rng)
    if n_samples - nq > 0:
        sev[~en_q] = rng.lognormal(mu, sigma, size=n_samples - nq)
    return sev

def calculer_hill(data, k_max):
    """Calcule l'estimateur de Hill pour chaque k de 2 à k_max."""
    data_tri = np.sort(data)
    n = len(data_tri)
    
    xi_hill = np.zeros(k_max)
    ic_bas = np.zeros(k_max)
    ic_haut = np.zeros(k_max)
    
    # Pre-calcul des logs pour optimiser
    log_data = np.log(data_tri)
    
    for k in range(2, k_max + 1):
        # Les k plus grandes valeurs
        log_exces = log_data[n-k : n]
        # Le seuil (la k+1 ème plus grande valeur)
        log_seuil = log_data[n-k-1]
        
        # Estimateur de Hill
        hill_val = np.mean(log_exces - log_seuil)
        xi_hill[k-1] = hill_val
        
        # Intervalle de confiance asymptotique à 95%
        erreur_std = hill_val / np.sqrt(k)
        ic_bas[k-1] = hill_val - 1.96 * erreur_std
        ic_haut[k-1] = hill_val + 1.96 * erreur_std
        
    return xi_hill, ic_bas, ic_haut

def main():
    print("Génération des données en cours...")
    rng = np.random.default_rng(42)
    donnees = generer_donnees_test(50000, rng)
    
    k_max = 3000 # On regarde jusqu'aux 3000 plus gros sinistres
    print("Calcul des estimateurs de Hill...")
    xi_hill, ic_bas, ic_haut = calculer_hill(donnees, k_max)
    
    # --- Création du Graphique ---
    plt.figure(figsize=(10, 6))
    
    # Tracer la courbe (on ignore k=0 et k=1)
    k_vals = np.arange(2, k_max + 1)
    plt.plot(k_vals, xi_hill[1:], color="#1E3461", linewidth=2, label="Estimateur de Hill ($\hat{\\xi}_k$)")
    
    # Tracer l'intervalle de confiance
    plt.fill_between(k_vals, ic_bas[1:], ic_haut[1:], color="#1E3461", alpha=0.15, label="IC à 95%")
    
    # Ligne de référence
    plt.axhline(y=XI_VRAI, color="#2563EB", linestyle="--", linewidth=2, label=f"Vrai paramètre $\\xi = {XI_VRAI}$")
    
    # Mise en forme
    plt.title("Graphique de Hill (Validation de l'indice de queue cyber)", fontsize=14, fontweight="bold", color="#111317")
    plt.xlabel("Nombre de sinistres extrêmes retenus ($k$)", fontsize=12)
    plt.ylabel("Indice de queue estimé ($\\xi$)", fontsize=12)
    plt.ylim(0.5, 2.5) # Focus sur la zone d'intérêt
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", fontsize=11)
    
    # Annotation de la zone de stabilité
    plt.text(1000, 2.2, "Zone de biais croissant\n(Pollution par le corps)", fontsize=10, ha="center", color="#555555")
    plt.text(150, 2.2, "Forte\nvariance", fontsize=10, ha="center", color="#555555")
    
    plt.tight_layout()
    plt.savefig("hill_plot_memoire.png", dpi=300)
    print("Le graphique a été sauvegardé sous 'hill_plot_memoire.png'.")
    plt.show()

if __name__ == "__main__":
    main()


