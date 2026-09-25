import numpy as np
import pandas as pd
import pytest

from pharmviginet.models.disproportionality import bcpnn, ebgm, fit_mgps_prior, pair_counts

PRIOR = (0.2, 0.1, 2.0, 4.0, 1 / 3)   # DuMouchel's default starting values


def _toy():
    rows = [(i, "D", "E") for i in (1, 2)] + [(i, "D", "OTHER") for i in (3, 4)]
    rows += [(5, "X", "E")] + [(i, "X", "OTHER") for i in range(6, 11)]
    rows += [(1, "D", "E")]  # duplicate
    return pd.DataFrame(rows, columns=["primaryid", "drugname", "pt"])


def test_pair_counts():
    c = pair_counts(_toy()).set_index(["drugname", "pt"])
    row = c.loc[("D", "E")]
    assert (row["a"], row["n_drug"], row["n_reac"], row["N"]) == (2, 4, 3, 10)
    assert row["E"] == pytest.approx(4 * 3 / 10)


def test_bcpnn_hand_computed():
    ic, ic025 = bcpnn([2], [1.2])
    assert ic[0] == pytest.approx(np.log2(2.5 / 1.7))
    assert ic025[0] == pytest.approx(ic[0] - 3.3 * 2.5 ** -0.5 - 2 * 2.5 ** -1.5)


def test_ebgm_shrinks_rare_pairs():
    g, g05 = ebgm(np.array([1.0, 500.0]), np.array([0.01, 5.0]), PRIOR)
    assert g[0] < 10          # raw ratio a/E = 100, one report → heavy shrinkage
    assert g[1] == pytest.approx(100, rel=0.1)   # lots of evidence → near raw ratio
    assert np.all(g05 < g)


def test_fit_mgps_prior_recovers_mean():
    rng = np.random.default_rng(0)
    a1, b1, a2, b2, p = 0.5, 0.2, 3.0, 3.0, 0.2
    n = 200_000
    E = rng.lognormal(mean=-1.0, sigma=1.5, size=n)
    comp1 = rng.random(n) < p
    lam = np.where(comp1, rng.gamma(a1, 1 / b1, n), rng.gamma(a2, 1 / b2, n))
    a = rng.poisson(lam * E)
    keep = a >= 1
    fit = fit_mgps_prior(a[keep], E[keep])
    f1, g1, f2, g2, fp = fit
    true_mean = p * a1 / b1 + (1 - p) * a2 / b2
    fit_mean = fp * f1 / g1 + (1 - fp) * f2 / g2
    assert fit_mean == pytest.approx(true_mean, rel=0.25)
