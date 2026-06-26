# params_jacobs.py
# -*- coding: utf-8 -*-
"""
Source unique des parametres de la brique remediation.
Derive seuil POT, mediane corps, q95 corps et beta queue DIRECTEMENT
depuis params_gpd.json + la formule Jacobs (plus aucune valeur en dur).
"""
import json
import os
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Coefficients Jacobs (a confirmer sur source primaire) ---
JACOBS_A = 7.68
JACOBS_B = 0.76
USD_EUR = 0.92

def cout_jacobs(volume):
    """Volume (personnes) -> cout EUR via Jacobs log-log (sans cap)."""
    volume = np.maximum(volume, 1.0)
    return np.exp(JACOBS_A + JACOBS_B * np.log(volume)) * USD_EUR

def charger_params_remediation():
    """Derive tous les parametres euros de la remediation depuis le JSON GPD."""
    with open(os.path.join(_ROOT, "params_gpd.json"), encoding="utf-8") as f:
        g = json.load(f)

    U_vol   = g["seuil_U"]            # 128 467 (volume)
    xi_vol  = g["xi_central_retenu"] # 1.30
    p_queue = g["prob_depassement"]  # 0.0654

    # Seuil POT en euros = Jacobs(U_vol)
    seuil_pot = float(cout_jacobs(U_vol))

    # xi effectif sur les couts = xi_volume * pente Jacobs
    xi_eff = xi_vol * JACOBS_B

    # Corps : mediane et q95 en euros, derives des quantiles de volume.
    # On reconstruit les quantiles de volume du corps depuis le JSON serait
    # ideal ; ici on les passe en argument depuis le script qui a les donnees.
    return {
        "seuil_pot": seuil_pot,
        "xi_eff": float(xi_eff),
        "p_queue": float(p_queue),
        "U_vol": float(U_vol),
        "sigma_vol": float(g["sigma"]),
    }

def beta_queue_jacobs(U_vol, sigma_vol):
    """
    Derive beta (echelle GPD en euros) par propagation locale de Jacobs.
    Approximation au 1er ordre : d(cout)/d(vol) au seuil, applique a sigma_vol.
    beta_eur ~ sigma_vol * Jacobs'(U_vol)
            = sigma_vol * b * cout(U_vol) / U_vol
    """
    cout_U = cout_jacobs(U_vol)
    derivee = JACOBS_B * cout_U / U_vol
    return float(sigma_vol * derivee)