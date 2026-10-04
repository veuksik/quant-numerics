import numpy as np
import pytest

from quant_numerics.pricing import (
    bs_european_call,
    bs_european_put,
    mc_european_call,
    mc_european_put,
)

ARGS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

PRICERS = [
    pytest.param(bs_european_call, mc_european_call, id="call"),
    pytest.param(bs_european_put, mc_european_put, id="put"),
]


def test_bs_call_known_value():
    assert bs_european_call(**ARGS) == pytest.approx(10.4506, abs=1e-3)


def test_put_call_parity():
    lhs = bs_european_call(**ARGS) - bs_european_put(**ARGS)
    rhs = ARGS["S"] - ARGS["K"] * np.exp(-ARGS["r"] * ARGS["T"])
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize(("bs", "mc"), PRICERS)
def test_mc_matches_black_scholes_within_3_se(bs, mc):
    price, se = mc(**ARGS, n=200_000, seed=42)
    assert abs(price - bs(**ARGS)) < 3 * se


def test_mc_parity_with_common_random_numbers():
    n = 200_000
    call, _ = mc_european_call(**ARGS, n=n, seed=42)
    put, _ = mc_european_put(**ARGS, n=n, seed=42)
    S, K, T, r, sigma = (ARGS[k] for k in ("S", "K", "T", "r", "sigma"))
    # Std error of the discounted terminal price: S * sqrt(exp(sigma^2 T) - 1) / sqrt(n)
    se_st = S * np.sqrt(np.exp(sigma**2 * T) - 1) / np.sqrt(n)
    assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=4 * se_st)


@pytest.mark.parametrize("mc", [mc_european_call, mc_european_put])
def test_mc_is_reproducible_for_fixed_seed(mc):
    assert mc(**ARGS, n=1_000, seed=7) == mc(**ARGS, n=1_000, seed=7)


@pytest.mark.parametrize("mc", [mc_european_call, mc_european_put])
def test_mc_rejects_n_below_2(mc):
    with pytest.raises(ValueError):
        mc(**ARGS, n=1, seed=0)


@pytest.mark.parametrize("mc", [mc_european_call, mc_european_put])
def test_mc_accepts_n_equal_2(mc):
    price, se = mc(**ARGS, n=2, seed=0)
    assert np.isfinite(price)
    assert np.isfinite(se)


@pytest.mark.parametrize(("bs", "mc"), PRICERS)
def test_standard_error_is_calibrated(bs, mc):
    """z = (price - exact) / se should behave like N(0, 1) across seeds."""
    exact = bs(**ARGS)
    z = np.empty(200)
    for s in range(200):
        price, se = mc(**ARGS, n=20_000, seed=s)
        z[s] = (price - exact) / se

    assert abs(z.mean()) < 0.3, f"mean(z) = {z.mean():.3f}"
    assert 0.8 < z.std(ddof=1) < 1.2, f"std(z) = {z.std(ddof=1):.3f}"
