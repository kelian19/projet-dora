# -*- coding: utf-8 -*-
"""
=============================================================================
COPULE DE GUMBEL — échantillonnage + agrégation des briques
=============================================================================
Relie les briques par une dépendance de queue supérieure (co-occurrence des
coûts extrêmes). theta=1 -> indépendance ; theta -> inf -> comonotonie.
Tau de Kendall = 1 - 1/theta (sert de test de validation de l'échantillonneur).

Deux méthodes d'agrégation :
  - paramétrique (Marshall-Olkin) : tire d'une vraie copule de Gumbel (référence)
  - par rangs : réordonne les marginales selon les rangs de la copule
                (préserve exactement les marginales)
"""

import numpy as np
from scipy import stats

NIVEAU_VAR = 0.995


def echantillon_gumbel(M, d, theta, rng):
    """
    Tire M points d'une copule de Gumbel dim d, paramètre theta (Marshall-Olkin).
    Variable de mélange = loi stable positive d'indice alpha=1/theta
    (méthode de Chambers-Mallows-Stuck pour stable positive asymétrique).
    """
    if theta <= 1.0 + 1e-9:
        return rng.random((M, d))

    alpha = 1.0 / theta
    U_ = rng.uniform(-np.pi / 2, np.pi / 2, size=M)
    E = rng.exponential(1.0, size=M)
    b = np.pi / 2
    terme1 = np.sin(alpha * (U_ + b)) / (np.cos(U_) ** (1.0 / alpha))
    terme2 = (np.cos(U_ - alpha * (U_ + b)) / E) ** ((1.0 - alpha) / alpha)
    S = (terme1 * terme2)[:, None]

    Ej = rng.exponential(1.0, size=(M, d))
    U = np.exp(-(Ej / S) ** alpha)
    return np.clip(U, 1e-12, 1 - 1e-12)


def make_quantile_empirique(echantillon):
    """Fonction quantile (inverse CDF) empirique d'un échantillon."""
    ech_trie = np.sort(echantillon)
    n = len(ech_trie)

    def q(u):
        idx = np.clip((u * n).astype(int), 0, n - 1)
        return ech_trie[idx]
    return q


def agreger_parametrique(marginales, theta, rng):
    """Agrège par échantillonnage paramétrique d'une copule de Gumbel."""
    M = len(marginales[0])
    d = len(marginales)
    U = echantillon_gumbel(M, d, theta, rng)
    L = np.zeros(M)
    for j in range(d):
        L += make_quantile_empirique(marginales[j])(U[:, j])
    return L


def agreger_par_rangs(marginales, theta, rng):
    """Agrège par réordonnancement des rangs (marginales préservées exactement)."""
    M = len(marginales[0])
    d = len(marginales)
    U = echantillon_gumbel(M, d, theta, rng)
    L = np.zeros(M)
    for j in range(d):
        marg_triee = np.sort(marginales[j])
        rangs = np.argsort(np.argsort(U[:, j]))
        L += marg_triee[rangs]
    return L


def valider_copule(rng):
    """Vérifie tau empirique = 1 - 1/theta (validation de l'échantillonneur)."""
    from scipy.stats import kendalltau
    print("=" * 60)
    print(" VALIDATION ECHANTILLONNEUR GUMBEL (tau = 1 - 1/theta)")
    print("=" * 60)
    ok_global = True
    for theta in [1.0, 1.5, 2.0, 3.0, 5.0]:
        U = echantillon_gumbel(50_000, 2, theta, rng)
        tau_emp, _ = kendalltau(U[:5000, 0], U[:5000, 1])
        tau_theo = 1.0 - 1.0 / theta
        ok = abs(tau_emp - tau_theo) < 0.05
        ok_global &= ok
        print(f"   theta={theta:.1f} | théo={tau_theo:.3f} | emp={tau_emp:.3f} "
              f"| {'OK' if ok else 'ECHEC'}")
    print(f"   => {'VALIDE' if ok_global else 'INCORRECT'}")
    print("=" * 60)
    return ok_global


if __name__ == "__main__":
    rng = np.random.default_rng(2024)
    valider_copule(rng)
