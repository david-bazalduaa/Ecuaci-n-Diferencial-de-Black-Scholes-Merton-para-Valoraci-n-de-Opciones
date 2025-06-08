import numpy as np
from scipy.stats import norm


def simulate_gbm(S0, mu, sigma, T, N, n_paths):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    
    dW = np.random.randn(n_paths, N) * np.sqrt(dt)

    for i in range(n_paths):
        W = np.cumsum(dW[i])
        paths[i, 1:] = S0 * np.exp((mu - 0.5 * sigma ** 2) * t[1:] + sigma * W)

    return t, paths


def valor_put(ST, K, T, r):
    payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)


def binaria_montecarlo(ST, K, T, r):
    payoff = (ST < K).astype(float)
    return np.exp(-r * T) * np.mean(payoff)

def black_scholes_put(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -K * np.exp(-r * T) * norm.cdf(d2) + S0 * norm.cdf(d1)
    
