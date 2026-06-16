# Projet DORA — Quantification du SCR risque opérationnel (Solvabilité 2)

Chaîne LDA complète pour quantifier le SCR_DORA : décomposition en briques,
calibration EVT, agrégation par copule de Gumbel, allocation d'Euler, et les
deux approches de périmètre (A = taxonomie/socle, C = contrefactuel/ouverture).

## Ordre d'exécution

Les modules de base n'ont aucune dépendance entre eux et peuvent être testés
isolément. Les deux scripts principaux les importent.

### 1. Modules de base (briques + outils)
- `brique_sanction.py`        — proxy réglementaire Beta x (2% du CA)
- `brique_prestataire.py`     — scénarios experts (3 quantiles)
- `brique_aggravation.py`     — contrefactuel lognormal
- `brique_remediation_corrigee.py` — corps lognormal + queue GPD/POT
- `copule_gumbel_agregation.py` — copule de Gumbel + validation Kendall
- `allocation_euler_var.py`   — allocation d'Euler-VaR (voisinage quantile)
- `ancrage_bilan_plausibilite.py` — profil bilanciel + test de plausibilité

Chacun se lance seul pour vérification :
```
python brique_sanction.py
python copule_gumbel_agregation.py   # lance la validation de Kendall
python ancrage_bilan_plausibilite.py
```

### 2. Scripts principaux (résultats)
- `scr_dora_synthese_finale.py` — APPROCHE A : SCR_DORA de référence,
  sensibilité à xi et au plafond, test de plausibilité.
```
python scr_dora_synthese_finale.py
```

- `approche_c_contrefactuel.py` — APPROCHE C : contrefactuel conforme vs
  non-conforme, Delta_DORA, décomposition fréquence/dépendance.
```
python approche_c_contrefactuel.py
```

## Résultats attendus (graine 42, M=500 000)

Approche A (scénario central, plafond 50 M€, xi=1,3) :
  SCR_DORA = 76,7 M€  -> PLAUSIBLE  (2,4x la charge formule standard ~32 M€)

Approche C (contrefactuel) :
  Non conforme : 116,6 M€  |  Conforme : 87,3 M€
  Delta_DORA = 29,3 M€ (borne supérieure, dire d'expert)

## Garde-fous (tests automatiques intégrés)
- copule : tau de Kendall empirique = 1 - 1/theta
- allocation : somme des contributions = SCR global ; alloué <= stand-alone
- plausibilité : SCR <= 0,3*BSCR, < CA, < fonds propres

## Points à assumer en soutenance
1. xi repris de la littérature (queue trop creuse pour estimer) -> sensibilité.
2. Fréquence lambda ramenée à UNE entité (pas le marché PRC entier).
3. Plafond de sévérité = choix de jugement (réassurance cyber) -> courbe SCR(plafond).
4. Approche C = ouverture critique (contrefactuel non observable), pas le socle.

## Dépendances
numpy, scipy. (matplotlib seulement pour les graphiques optionnels.)

## Note sur les données PRC
La calibration de la remédiation a été ancrée sur la base PRC réelle
(Data_Breach_Chronology, incidents HACK). Les paramètres du corps
(médiane 171k€, q95 9,3M€, seuil POT 4,95M€) en sont issus. Pour recalibrer
sur une autre base, ajuster REM_MEDIANE_CORPS, REM_Q95_CORPS, REM_SEUIL_POT
dans scr_dora_synthese_finale.py et approche_c_contrefactuel.py.
