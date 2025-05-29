# README - Proyecto: Valoración de Opciones con la Ecuación de Black-Scholes-Merton
---

## Contenido del Proyecto

Se trabajará a partir de la ecuación diferencial parcial de Black-Scholes-Merton, que describe el precio de una opción financiera \( V(S, t) \) en función del precio del activo subyacente \( S \) y del tiempo \( t \):

\[
\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0
\]

donde:
- \( S(t) \): precio del activo subyacente
- \( V(S,t) \): valor de la opción
- \( \sigma \): volatilidad
- \( r \): tasa libre de riesgo


### Fase 1: Cambio de variable
- Transformación de la ecuación original mediante el cambio de variable `x = ln(S)`.
- Objetivo: Linealizar el comportamiento del precio del activo subyacente.

### Fase 2: Interpretación como sistema de control
- Modelado del sistema mediante funciones de transferencia o representación en espacio de estados.
- Construcción de diagramas de bloques incorporando retroalimentaciones y coeficientes clave (`r`, `σ`).
- Discretización del dominio espacial para aproximación por ODEs.

### Fase 3: Análisis de estabilidad
- Estudio de condiciones para la convergencia o divergencia del sistema.
- Análisis espectral de la versión linealizada.

### Fase 4: Estrategia de cobertura como control
- Construcción de un portafolio de cobertura con una opción y activos.
- Aplicación de la fórmula de Itô para deducir la dinámica del portafolio.
- Derivación de la ecuación de Black-Scholes-Merton a partir del portafolio sin riesgo.
- Desarrollo de una ley de control y análisis de su efecto sobre la estabilidad y respuesta del sistema.

### Fase 5: Simulación de Monte Carlo
- Simulación de trayectorias de precios bajo un marco riesgo-neutral.
- Valoración de opciones PUT y binarias mediante Monte Carlo.
- Comparación de resultados con la fórmula cerrada y cálculo del error según el número de iteraciones.

### Fase 6: Resolución numérica de la EDP
- Implementación del esquema de diferencias finitas implícito.
- Discretización temporal y espacial adecuada para opciones PUT europeas y binarias.
- Comparación entre los resultados obtenidos con el método numérico y la simulación de Monte Carlo.

---

## Entregables

- **Informe técnico completo**: Documento que describa detalladamente cada fase, incluyendo desarrollo teórico, implementación, resultados y análisis numérico.
- **Gráficos y tablas** que sustenten las conclusiones obtenidas en cada etapa.
- **Presentación final** con resumen del enfoque, metodología y resultados clave del proyecto.

---

## Bibliografía principal

- Black, F. & Scholes, M. (1973). *The pricing of options and corporate liabilities*. Journal of Political Economy, 81, 637–654.
- Merton, R. C. (1973). *Theory of rational option pricing*. The Bell Journal of Economics and Management Science, 4(1), 141–183.
- Hull, J. C. (2021). *Options, Futures and Other Derivatives*.
- Evans, L. (2024). *An Introduction to Mathematical Optimal Control Theory*. Disponible en: [https://math.berkeley.edu/~evans/control.course.pdf](https://math.berkeley.edu/~evans/control.course.pdf)
