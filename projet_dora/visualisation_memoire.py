# -*- coding: utf-8 -*-
"""
=============================================================================
VISUALISATION AVANCEE — SCR DORA & ALLOCATION D'EULER
=============================================================================
Génère les graphiques pour le mémoire et les présentations métier.
Intègre nativement la charte graphique et la typographie requises.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# =============================================================================
# CONFIGURATION DE LA CHARTE GRAPHIQUE
# =============================================================================
COLOR_BG = "#f6f7f9"
COLOR_TEXT = "#111317"
COLOR_PRIMARY = "#2563eb"
COLOR_SECONDARY = "#64748b" # Gris bleuté pour les éléments secondaires
COLOR_ALERT = "#ef4444"     # Rouge pour le plafond réglementaire

plt.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_TEXT,
    "axes.labelcolor": COLOR_TEXT,
    "text.color": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Fallback si les polices exactes ne sont pas installées sur l'OS
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Montserrat", "Arial"],
})


def plot_sensibilite_plafond():
    """Génère la courbe de sensibilité du SCR DORA au plafond individuel."""
    
    # Données issues de scr_dora_synthese_finale.py
    plafonds_m = np.array([20, 50, 75, 100, 196])
    scr_dora_m = np.array([54.0, 76.0, 91.9, 112.2, 198.6])
    plafond_reglementaire_m = 108.0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Trace la courbe principale
    ax.plot(plafonds_m, scr_dora_m, marker='o', markersize=8, linewidth=2.5, 
            color=COLOR_PRIMARY, label="SCR DORA (VaR 99.5%)")

    # Trace la limite de plausibilité réglementaire
    ax.axhline(y=plafond_reglementaire_m, color=COLOR_ALERT, linestyle='--', 
               linewidth=2, label="Plafond Opérationnel (0,3 * BSCR)")

    # Remplissage de la zone de bascule
    ax.fill_between(plafonds_m, scr_dora_m, plafond_reglementaire_m, 
                    where=(scr_dora_m > plafond_reglementaire_m), 
                    interpolate=True, color=COLOR_ALERT, alpha=0.1, 
                    label="Zone Non Plausible")

    # Esthétique et annotations
    ax.set_title("Sensibilité du SCR DORA au plafond de perte maximale", 
                 fontsize=16, fontweight="semibold", pad=20)
    ax.set_xlabel("Plafond de sévérité individuel (en M€)", fontsize=12)
    ax.set_ylabel("Capital Requis - SCR (en M€)", fontsize=12)
    
    ax.legend(loc="upper left", frameon=True, facecolor=COLOR_BG, edgecolor=COLOR_BG)
    ax.grid(True, alpha=0.6)

    plt.tight_layout()
    plt.savefig("sensibilite_plafond.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_allocation_euler():
    """Génère un graphique en barres comparant VaR Standalone et Allocation Euler."""
    
    # Données illustratives issues de l'agrégation
    briques = ["Remédiation", "Prestataire", "Sanction", "Aggravation"]
    var_standalone_m = np.array([61.76, 45.26, 5.70, 0.97])
    
    # Approximation de l'allocation d'Euler après diversification (~31%)
    euler_allocation_m = var_standalone_m * 0.69 

    x = np.arange(len(briques))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Barres
    bars1 = ax.bar(x - width/2, var_standalone_m, width, label='VaR Stand-alone', 
                   color=COLOR_SECONDARY)
    bars2 = ax.bar(x + width/2, euler_allocation_m, width, label="Contribution d'Euler", 
                   color=COLOR_PRIMARY)

    # Esthétique et annotations
    ax.set_title("Bénéfice de diversification et Allocation d'Euler", 
                 fontsize=16, fontweight="semibold", pad=20)
    ax.set_ylabel("M€", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(briques, fontsize=11)
    
    ax.legend(frameon=True, facecolor=COLOR_BG, edgecolor=COLOR_BG)
    ax.grid(axis='y', alpha=0.6)

    # Ajout des étiquettes de valeurs au-dessus des barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points de décalage vertical
                        textcoords="offset points",
                        ha='center', va='bottom', color=COLOR_TEXT, fontsize=10)

    plt.tight_layout()
    plt.savefig("allocation_euler.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_copule_gumbel_dependance():
    """
    Génère des nuages de points (scatter plots) pour illustrer la 
    dépendance de queue supérieure d'une copule de Gumbel selon le paramètre theta.
    """
    # Import de votre fonction d'échantillonnage existante
    try:
        from copule_gumbel_agregation import echantillon_gumbel
    except ImportError:
        print("Erreur : Assurez-vous que copule_gumbel_agregation.py est dans le même dossier.")
        return

    rng = np.random.default_rng(42)
    M = 2500  # Nombre de points réduit pour que le graphique reste lisible
    
    # Valeurs de theta à comparer 
    # 1.0 = Indépendance | 1.5 = Scénario central | 3.0 = Forte contagion systémique
    thetas = [1.0, 1.5, 3.0]
    titres = [
        "Indépendance ($\\theta = 1.0$)", 
        "Dépendance Modérée ($\\theta = 1.5$)", 
        "Forte Contagion ($\\theta = 3.0$)"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle("Illustration de la concentration des risques cyber extrêmes (Copule de Gumbel)", 
                 fontsize=16, fontweight="semibold", y=1.05)

    for i, ax in enumerate(axes):
        # Génération de l'échantillon (dimension 2 pour la visualisation)
        U = echantillon_gumbel(M, 2, thetas[i], rng)
        
        # Nuage de points avec la couleur de la charte
        ax.scatter(U[:, 0], U[:, 1], alpha=0.4, color=COLOR_PRIMARY, s=12, edgecolors='none')
        
        # Mise en évidence du quadrant extrême (les 5% pires cas)
        ax.axvline(x=0.95, color=COLOR_ALERT, linestyle=':', alpha=0.7)
        ax.axhline(y=0.95, color=COLOR_ALERT, linestyle=':', alpha=0.7)
        ax.fill_between([0.95, 1.0], 0.95, 1.0, color=COLOR_ALERT, alpha=0.1)
        
        # Esthétique
        ax.set_title(titres[i], fontsize=12, pad=10)
        ax.set_xlabel("Quantile Brique 1 (ex: Prestataire)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Quantile Brique 2 (ex: Remédiation)", fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("copule_gumbel_dependance.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_waterfall_contrefactuel():
    """
    Génère un diagramme en cascade (Waterfall) illustrant la décomposition 
    du coût en capital du non-respect de DORA (Delta DORA).
    """
    # Données issues de l'exécution de approche_c_contrefactuel.py
    etapes = [
        "SCR DORA\n(Non Conforme)", 
        "Effet Fréquence\n(Hygiène cyber)", 
        "Effet Dépendance\n(Stratégie de sortie)", 
        "SCR DORA\n(Conforme)"
    ]
    
    # Valeurs (hauteurs des barres)
    valeurs = [116.6, 14.3, 15.0, 87.3]
    
    # Bases des barres (pour l'effet de flottaison du waterfall)
    bases = [0, 102.3, 87.3, 0]
    
    # Attribution des couleurs (Rouge/Alerte pour le départ, Gris pour la baisse, Bleu pour la cible)
    couleurs = [COLOR_ALERT, COLOR_SECONDARY, COLOR_SECONDARY, COLOR_PRIMARY]

    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Positionnement sur l'axe X
    x = np.arange(len(etapes))
    largeur_barre = 0.55

    # Création des barres
    bars = ax.bar(x, valeurs, bottom=bases, color=couleurs, width=largeur_barre)

    # Ajout des lignes de liaison en pointillés entre les colonnes
    ax.plot([0, 1], [116.6, 116.6], color=COLOR_TEXT, linestyle='--', linewidth=1, alpha=0.4)
    ax.plot([1, 2], [102.3, 102.3], color=COLOR_TEXT, linestyle='--', linewidth=1, alpha=0.4)
    ax.plot([2, 3], [87.3, 87.3], color=COLOR_TEXT, linestyle='--', linewidth=1, alpha=0.4)

    # Étiquettes de données sur chaque barre
    for i, bar in enumerate(bars):
        y_val = bar.get_height() + bar.get_y()
        # On ajoute un signe "-" pour les effets de réduction
        texte = f"{valeurs[i]:.1f}" if i in [0, 3] else f"- {valeurs[i]:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, y_val + 1.5, texte, 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color=COLOR_TEXT)

    # Esthétique générale
    ax.set_title("Décomposition du bénéfice en capital de la conformité DORA", 
                 fontsize=16, fontweight="semibold", pad=20)
    ax.set_ylabel("Capital Requis - SCR (en M€)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(etapes, fontsize=11)
    
    # On ajuste l'axe Y pour laisser de la place au texte en haut
    ax.set_ylim(0, 130)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("waterfall_conformite.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Génération du graphique de sensibilité au plafond...")
    plot_sensibilite_plafond()
    
    print("Génération du graphique d'allocation d'Euler...")
    plot_allocation_euler()
    
    print("Génération du graphique de la copule de Gumbel...")
    plot_copule_gumbel_dependance()
    
    print("Génération du diagramme Waterfall (Approche C)...")
    plot_waterfall_contrefactuel()
    
    print("Tous les graphiques ont été sauvegardés avec succès !")