"""
Classical disproportionality statistics on FAERS report counts.

Per (drugname, pt) pair, from distinct reports (primaryid):
    a       reports with drug and event
    n_drug  reports with drug
    n_reac  reports with event
    N       all reports
    E       expected count under independence = n_drug * n_reac / N

Methods:
    PRR    proportional reporting ratio
    BCPNN  information component IC and lower bound IC025 (Norén et al. 2006)
    MGPS   EBGM and 5th percentile EB05 (DuMouchel 1999), 2-gamma mixture prior
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammainc, gammaln

PAIR = ["drugname", "pt"]


def pair_counts(df: pd.DataFrame) -> pd.DataFrame:
    """df: [primaryid, drugname, pt] rows (duplicates allowed) → counts per pair."""
    rep = df[["primaryid", "drugname", "pt"]].drop_duplicates()
    N = rep["primaryid"].nunique()
    a = rep.groupby(PAIR, observed=True).size().rename("a").reset_index()
    n_drug = rep.groupby("drugname", observed=True)["primaryid"].nunique().rename("n_drug").reset_index()
    n_reac = rep.groupby("pt", observed=True)["primaryid"].nunique().rename("n_reac").reset_index()
    c = a.merge(n_drug, on="drugname").merge(n_reac, on="pt")
    c["N"] = N
    c["E"] = c["n_drug"] * c["n_reac"] / N
    return c


def prr(c: pd.DataFrame) -> pd.Series:
    """PRR = (a / n_drug) / ((n_reac - a) / (N - n_drug)); NaN when undefined."""
    denom = (c["n_reac"] - c["a"]) / (c["N"] - c["n_drug"])
    return (c["a"] / c["n_drug"]) / denom.replace(0, np.nan)


def compute_prr(df: pd.DataFrame) -> pd.DataFrame:
    c = pair_counts(df)
    c["prr"] = prr(c)
    return c[PAIR + ["prr"]]


def bcpnn(a, E) -> tuple[np.ndarray, np.ndarray]:
    """IC = log2((a+0.5)/(E+0.5)); IC025 via Norén's closed-form approximation."""
    a = np.asarray(a, dtype=float)
    E = np.asarray(E, dtype=float)
    ic = np.log2((a + 0.5) / (E + 0.5))
    ic025 = ic - 3.3 * (a + 0.5) ** -0.5 - 2.0 * (a + 0.5) ** -1.5
    return ic, ic025


# ── MGPS ──────────────────────────────────────────────────────────────────────

def _nb_logpmf(a, E, alpha, beta):
    """log P(a) when a ~ Poisson(λE), λ ~ Gamma(alpha, rate=beta)."""
    return (gammaln(alpha + a) - gammaln(alpha) - gammaln(a + 1)
            + alpha * np.log(beta / (beta + E)) + a * np.log(E / (beta + E)))


def _unpack(theta):
    a1, b1, a2, b2 = np.exp(theta[:4])
    return a1, b1, a2, b2, expit(theta[4])


def _neg_loglik(theta, a, E, w):
    a1, b1, a2, b2, p = _unpack(theta)
    l1 = np.log(p) + _nb_logpmf(a, E, a1, b1)
    l2 = np.log1p(-p) + _nb_logpmf(a, E, a2, b2)
    # zero-truncated: only pairs with a >= 1 are observed
    z = p * (b1 / (b1 + E)) ** a1 + (1 - p) * (b2 / (b2 + E)) ** a2
    ll = np.logaddexp(l1, l2) - np.log1p(-np.minimum(z, 1 - 1e-12))
    return -np.sum(w * ll)


def fit_mgps_prior(a, E, init=(0.2, 0.1, 2.0, 4.0, 1 / 3)) -> tuple[float, ...]:
    """
    Fit (alpha1, beta1, alpha2, beta2, p) by maximum marginal likelihood.
    Pairs are squashed on (a, E rounded to 3 significant digits) for speed.
    """
    E = np.asarray(E, dtype=float)
    mag = 10.0 ** (np.floor(np.log10(E)) - 2)
    E_sq = np.round(E / mag) * mag
    sq = (pd.DataFrame({"a": np.asarray(a, dtype=float), "E": E_sq})
          .groupby(["a", "E"]).size().reset_index(name="w"))
    x0 = np.r_[np.log(init[:4]), np.log(init[4] / (1 - init[4]))]
    res = minimize(_neg_loglik, x0, args=(sq["a"].values, sq["E"].values, sq["w"].values),
                   method="L-BFGS-B")
    return tuple(float(v) for v in _unpack(res.x))


def _posterior_q(a, E, prior):
    a1, b1, a2, b2, p = prior
    l1 = np.log(p) + _nb_logpmf(a, E, a1, b1)
    l2 = np.log1p(-p) + _nb_logpmf(a, E, a2, b2)
    return np.exp(l1 - np.logaddexp(l1, l2))


def ebgm(a, E, prior, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """EBGM = exp(E[log λ | a]) and the q-quantile of λ | a (EB05 for q=0.05)."""
    a = np.asarray(a, dtype=float)
    E = np.asarray(E, dtype=float)
    a1, b1, a2, b2, _ = prior
    Q = _posterior_q(a, E, prior)
    s1, r1, s2, r2 = a1 + a, b1 + E, a2 + a, b2 + E
    elog = Q * (digamma(s1) - np.log(r1)) + (1 - Q) * (digamma(s2) - np.log(r2))

    # bisection on log λ for the mixture CDF
    lo = np.full_like(a, -30.0)
    hi = np.full_like(a, 30.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        x = np.exp(mid)
        cdf = Q * gammainc(s1, r1 * x) + (1 - Q) * gammainc(s2, r2 * x)
        below = cdf < q
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.exp(elog), np.exp((lo + hi) / 2)
