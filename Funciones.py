#Importo las librerias
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Definición de funciones para simulación de Monte Carlo y Black-Scholes

#Función para simular un movimiento browniano geométrico (GBM)
def simulate_gbm(S0, mu, sigma, T, N, n_paths):
    dt = T / N
    # t es el vector de tiempos, paths es la matriz de trayectorias
    t = np.linspace(0, T, N + 1)
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0
    np.random.seed(42)  #Para reproducibilidad
    # Generar incrementos del proceso de Wiener
    # dW es el incremento del proceso de Wiener
    dW = np.random.randn(n_paths, N) * np.sqrt(dt)

    # Calcular las trayectorias del GBM
    for i in range(n_paths):
        W = np.cumsum(dW[i])
        paths[i, 1:] = S0 * np.exp((mu - 0.5 * sigma ** 2) * t[1:] + sigma * W)

    # Retornar el vector de tiempos y la matriz de trayectorias
    return t, paths

#Función para calcular el valor de una opción put europea usando Monte Carlo
def valor_put(ST, K, T, r):
    payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

#Función para calcular el valor de una opción binaria put europea usando Monte Carlo
def binaria_montecarlo(ST, K, T, r):
    payoff = (ST < K).astype(int)
    return np.exp(-r * T) * np.mean(payoff)

#Función para calcular el valor de una opción put europea usando Black-Scholes
def black_scholes_put(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

#Función para calcular el valor de una opción binaria put europea usando Black-Scholes
def binaria_black_scholes(S0, K, T, r, sigma):
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    binario_put = np.exp(-r * T) * norm.cdf(-d2)
    return binario_put

#Función para inicializar la malla de precios y tiempos
def inicializar_malla(M, K,S_max, tipo='put'):
    V=np.zeros(M+1)  # Malla de precios y tiempos
    dS = S_max / M  # Tamaño del paso en la malla de precios
    # Inicializar los valores de la malla según el tipo de opción
    for i in range(M+1):
        Si = i * dS
        if tipo == 'put':
            V[i] = max(K - Si, 0)
        elif tipo == 'binario':
            V[i] = 1 if Si < K else 0
    return V


#Función para crear la matriz tridiagonal para el método implícito
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
    

#Función para resolver un sistema tridiagonal 
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

#Función para resolver el sistema implícito para opciones europeas usando el método implícito
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


#Función para calcular errores entre el valor de referencia y el valor calculado por el método implícito
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

    # Obtener listas separadas para M y errores
    M_vals = [M for M, _, _, _, _ in errores]
    error_vals = [err for _, _, _, _, err in errores]
    
    plt.figure(figsize=(10, 6))
    plt.plot(M_vals, error_vals, marker='o')
    plt.xlabel("Número de subdivisiones M (=N)")
    plt.ylabel("Error absoluto")
    plt.title("Error absoluto vs tamaño de malla")
    plt.grid(True, which='both')
    plt.show()
    
    return errores


#Función para comparar la solución implícita con Monte Carlo
def comparar_soluciones(S0, K, r, sigma, T, valoresMN, S_max,M,N, n_paths=1000000, tipo='put'):
    print(f"\nComparación entre Método Implícito y Monte Carlo")
    print(f"Parámetros: S0={S0}, K={K}, r={r}, σ={sigma}, T={T}, M=N={M}, paths={n_paths}")

    # Valor Black-Scholes como referencia
    if tipo == 'put':
        valor_bs = black_scholes_put(S0, K, T, r, sigma)
    else:
        valor_bs = binaria_black_scholes(S0, K, T, r, sigma)

    # Simulación de trayectorias Monte Carlo
    _, paths = simulate_gbm(S0, r, sigma, T, N, n_paths)
    ST = paths[:, -1]  # Valor del activo al tiempo T

    # Monte Carlo
    if tipo == 'put':
        valor_mc = valor_put(ST, K, T, r)
    elif tipo == 'binario':
        valor_mc = binaria_montecarlo(ST, K, T, r)

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

    print(f"\n tipo de opción: {tipo}")
    print(f"Black-Scholes: {valor_bs:.6f}")
    print(f"Monte Carlo: {valor_mc:.6f}")
    print(f"Método Implícito: {V:.6f}")
    print(f"Error absoluto: {abs(valor_mc - V):.6f}")

    # Gráfico comparativo
    etiquetas = ['Black-Scholes', 'Monte Carlo', 'Implícito']
    valores = [valor_bs, valor_mc, V]

    plt.figure(figsize=(8, 5))
    plt.bar(etiquetas, valores, color=['skyblue', 'orange', 'green'])
    plt.title(f"Comparación de métodos para opción {tipo}")
    plt.ylabel("Valor de la opción")
    plt.grid(axis='y')
    plt.show()

    return valor_bs, valor_mc, V