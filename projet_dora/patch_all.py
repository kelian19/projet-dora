# -*- coding: utf-8 -*-
"""Patch les 3 fichiers restants en une seule passe."""
import os

# ── 1. sensibilite_delta_dora.py ────────────────────────────────────────
f = 'projet_dora/sensibilite_delta_dora.py'
c = open(f, encoding='utf-8').read()
c = c.replace('PLAFOND_INDIV = 50_000_000.0',
              'PLAFOND_INDIV = 40_000_000.0   # plafond central 40 M, aligne approche A')
open(f, 'w', encoding='utf-8').write(c)
print('sensibilite_delta_dora   :', 'OK' if '40_000_000' in c else 'ECHEC')

# ── 2. scr_dora_synthese_finale.py ──────────────────────────────────────
f = 'projet_dora/scr_dora_synthese_finale.py'
c = open(f, encoding='utf-8').read()
c = c.replace('CHOIX CENTRAL : plafond = 50 M',
              'CHOIX CENTRAL : plafond = 40 M')
c = c.replace('(~10% des fonds propres',
              '(~5% des fonds propres')
c = c.replace('plafond 50 M',
              'plafond 40 M')
c = c.replace('PLAFOND_CENTRAL = 50_000_000.0',
              'PLAFOND_CENTRAL = 40_000_000.0')
open(f, 'w', encoding='utf-8').write(c)
print('scr_dora_synthese_finale :', 'OK' if 'CHOIX CENTRAL : plafond = 40 M' in c else 'ECHEC')

# ── 3. bootstrap_incertitude_scr.py ─────────────────────────────────────
f = 'projet_dora/bootstrap_incertitude_scr.py'
c = open(f, encoding='utf-8').read()

old = '    return {"xi": xi, "lam": lam, "facteur": facteur, "theta": theta, "q95": q95}'
new = '''    alpha_s = rng.uniform(0.5, 1.5)
    beta_s  = rng.uniform(4.0, 9.0)
    proba_s = rng.uniform(0.05, 0.20)
    scenarios_ref = [
        {"nom": "Panne hyperscaler cloud",            "proba_an": 0.08,
         "q50": 2_000_000, "q95": 25_000_000,  "q995": 150_000_000, "loi": "gpd"},
        {"nom": "Compromission prestataire paiement", "proba_an": 0.05,
         "q50": 1_000_000, "q95": 12_000_000,  "q995":  80_000_000, "loi": "gpd"},
        {"nom": "Rancongiciel editeur metier",        "proba_an": 0.12,
         "q50":   300_000, "q95":  3_000_000,  "q995":  20_000_000, "loi": "lognorm"},
    ]
    import numpy as _np
    scenarios_tires = []
    for s in scenarios_ref:
        s_tire = dict(s)
        s_tire["q95"]  = s["q95"]  * _np.exp(rng.normal(0.0, 0.30))
        s_tire["q995"] = s["q995"] * _np.exp(rng.normal(0.0, 0.30))
        s_tire["q95"]  = max(s_tire["q95"],  s["q50"] * 1.01)
        s_tire["q995"] = max(s_tire["q995"], s_tire["q95"] * 1.01)
        scenarios_tires.append(s_tire)
    return {"xi": xi, "lam": lam, "facteur": facteur, "theta": theta, "q95": q95,
            "alpha_s": alpha_s, "beta_s": beta_s, "proba_s": proba_s,
            "scenarios_presta": scenarios_tires}'''

if old in c:
    c = c.replace(old, new)
    open(f, 'w', encoding='utf-8').write(c)
    print('bootstrap_incertitude_scr:', 'OK' if 'alpha_s' in c else 'ECHEC')
else:
    print('bootstrap_incertitude_scr: ECHEC — pattern non trouve')
    idx = c.find('def tirer_parametres')
    print('  -> extrait:', repr(c[idx:idx+300]))
