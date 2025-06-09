import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

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

def inicializar_malla(M, K,S_max, tipo='put'):
    V=np.zeros(M+1)  # Malla de precios y tiempos
    dS = S_max / M  # Tamaño del paso en la malla de precios
    for i in range(M+1):
        Si = i * dS
        if tipo == 'put':
            V[i] = max(K - Si, 0)
        elif tipo == 'binario':
            V[i] = 1 if Si < K else 0
    return V

def crear_matriz_tridiagonal(M, dS, dT, sigma, r):
    A = np.zeros((M-1, M-1))

    for i in range(1,M):
        Si=i * dS
        alpha= 0.5*dT*(sigma**2 * Si**2 / dS**2 - r * Si / dS)
        beta=1 + dT * (sigma**2 * Si**2 / dS**2 + r)
        gamma=0.5 * dT * (sigma**2 * Si**2 / dS**2 + r * Si / dS)

        if i > 1:
            A[i-1, i-2] = -alpha  # subdiagonal
        A[i-1, i-1] = beta       # diagonal
        if i < M-1:
            A[i-1, i] = -gamma   # superdiagonal
    return A
    

def sol_tridiagonal(A, b):
    n= len(b)
    cp= np.zeros(n)
    dp= np.zeros(n)

    cp[0] = A[0][1] / A[0][0]
    dp[0] = b[0] / A[0][0]

    for i in range(1, n-1):
        m = A[i][i] - A[i][i-1] * cp[i-1]
        cp[i] = A[i][i+1] / m
        dp[i] = (b[i] - A[i][i-1] * dp[i-1]) / m

    m = A[n-1][n-1] - A[n-1][n-2] * cp[n-2]
    dp[n-1] = (b[n-1] - A[n-1][n-2] * dp[n-2]) / m

    x = np.zeros(n)
    x[n-1] = dp[n-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x


def solucion_implicita(dT,A,V,K,r,dS,S0,T,N,tipo='put'):
    for j in range(N-1, 0, -1):
        t = j * dT

        # RESOLVER el sistema para nodos internos
        V_interior = sol_tridiagonal(A, V[1:-1])

        # ACTUALIZAR nodos internos
        V[1:-1] = V_interior

        if tipo == 'put':
            V[0] = K * np.exp(-r * (T - t))
            V[-1] = 0
        elif tipo == 'binario':
            V[0] = np.exp(-r * (T - t))  # el valor presente del pago 1
            V[-1] = 0


    index = S0 / dS
    i = int(index)
    fraccion = index - i
    if i >= len(V)-1:
        return V[-1]
    V_interp = (1-fraccion)*V[i] + fraccion*V[i+1]
    return V_interp



def calcular_errores(S0, K, r, sigma, T, valoresMN,S_max,tipo='put'):
    referencia=black_scholes_put(S0, K, T, r, sigma) if tipo == 'put' else binaria_black_scholes(S0, K, T, r, sigma)
    errores = []
    for M, N in valoresMN:
        dS=S_max/M  # Tamaño del paso en la malla de precios
        dT = T/N  # Tamaño del paso en la malla de tiempo
        V= inicializar_malla(M, K, S_max, tipo)
        #Añadimos las condiciones de frontera
        if tipo == 'put':
            V[0]=K*np.exp(-r*T)  # Condición de frontera inferior para PUT
            V[-1] = 0  # Condición de frontera superior para PUT

        elif tipo == 'binario':
            V[0]=np.exp(-r*T)  # Condición de frontera inferior para binario
            V[-1] = 0  # Condición de frontera superior para binario

        #Matriz tridiagonal
        A = crear_matriz_tridiagonal(M, dS, dT, sigma, r)
        V = solucion_implicita(dT,A,V,K,r,dS,S0,T,N,tipo=tipo)
        
        error = abs(referencia - V)
        errores.append((M, N, V, referencia, error))
    
    #plt.figure()
    #plt.loglog(valoresMN[0], [tupla[-1] for tupla in errores], marker='o')
    #plt.xlabel("Número de subdivisiones M (=N)")
    #plt.ylabel("Error absoluto")
    #plt.title("Error absoluto vs tamaño de malla (log-log)")
    #plt.grid(True)
    #plt.show()
    
    return errores
