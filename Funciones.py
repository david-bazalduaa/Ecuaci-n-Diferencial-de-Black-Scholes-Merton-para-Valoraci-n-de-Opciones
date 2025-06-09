import numpy as np
from scipy.stats import norm
from scipy.sparse.linalg import spsolve


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
    payoff = (ST < K).astype(int)
    return np.exp(-r * T) * np.mean(payoff)

def black_scholes_put(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -K * np.exp(-r * T) * norm.cdf(d2) + S0 * norm.cdf(d1)

def binaria_black_scholes(S0, K, T, r, sigma):
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    binario_put = np.exp(-r * T) * norm.cdf(-d2)
    return binario_put

    

def sol_put(M, V, alpha,K, gamma, A, S0, S,N, tipo='put'):
    if tipo=='binario':
        V[:, -1] = np.where(S < K, 1, 0)

    # Resolución backward in time
    for j in reversed(range(N)):
        # Vector del lado derecho
        b = V[1:-1, j+1].copy()
        # Añadir condiciones de frontera
        b[0] += alpha[0] * V[0, j]
        b[-1] += gamma[-1] * V[-1, j]
        # Resolver sistema
        V[1:-1, j] = spsolve(A, b)

    # Interpolar para obtener el precio en S0
    put_price = np.interp(S0, S, V[:, 0])
    return put_price, V

def calcular_errores(S0, K, r, sigma, T, valoresMN,V,alpha,gamma, A, S,tipo='put'):
    referencia=black_scholes_put(S0, K, T, r, sigma) if tipo == 'put' else binaria_black_scholes(S0, K, T, r, sigma)
    errores = []
    for M, N in valoresMN:
        num = sol_put(M, V, alpha,K, gamma, A, S0, S,N, tipo)
        error = abs(num - referencia)
        errores.append((M, N, num, referencia, error))
    
    return errores
