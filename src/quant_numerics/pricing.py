"""European option pricing: closed-form Black-Scholes and Monte Carlo.

Conventions: S spot, K strike, T time to expiry in years, r continuously
compounded annual risk-free rate, sigma annualized volatility.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """Return d1 and d2 of the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def _simulate_terminal_prices(S, T, r, sigma, n, seed):
    """Simulate n terminal prices under GBM with risk-neutral drift."""
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def _mean_and_se(discounted_payoffs):
    """Return the sample mean and its standard error."""
    n = len(discounted_payoffs)
    return discounted_payoffs.mean(), discounted_payoffs.std(ddof=1) / np.sqrt(n)


def bs_european_call(S, K, T, r, sigma):
    """Black-Scholes price of a European call."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_european_put(S, K, T, r, sigma):
    """Black-Scholes price of a European put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def mc_european_call(S, K, T, r, sigma, n, seed):
    """Monte Carlo price of a European call.

    Returns:
        (price, standard_error)
    """
    ST = _simulate_terminal_prices(S, T, r, sigma, n, seed)
    return _mean_and_se(np.exp(-r * T) * np.maximum(ST - K, 0.0))


def mc_european_put(S, K, T, r, sigma, n, seed):
    """Monte Carlo price of a European put.

    Returns:
        (price, standard_error)
    """
    ST = _simulate_terminal_prices(S, T, r, sigma, n, seed)
    return _mean_and_se(np.exp(-r * T) * np.maximum(K - ST, 0.0))
