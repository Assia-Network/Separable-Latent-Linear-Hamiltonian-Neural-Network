# SLL-HNN: Separable Latent Linear Hamiltonian Neural Networks para el Sistema Solar

## 1. Motivación

Este proyecto nace de una profunda curiosidad personal por comprender la física del universo y mi fascinación por el desarrollo de modelos de *Machine Learning* informados por la física. Explorar el cosmos a través del código siempre ha sido un objetivo para mí, pero al abordar el problema gravitacional de N-cuerpos en el Sistema Solar, me encontré con una brecha crítica en la mecánica computacional actual [[1]](#referencias).

Aunque el aprendizaje profundo ha revolucionado la aproximación de funciones, descubrimos que los modelos convencionales ignoran el rigor matemático que exigen los sistemas celestes [[1]](#referencias). Por ello, el desarrollo de la arquitectura **SLL-HNN** está impulsado por dos necesidades fundamentales:

* **Preservación Estricta de Invariantes Físicos:** La necesidad imperativa de respetar la coherencia dimensional y preservar la estructura simpléctica para evitar que el sistema disipe energía artificialmente durante simulaciones a escalas de tiempo astronómicas [[1]](#referencias).
* **Superación de Límites Clásicos (El Problema del Mezclado):** Las Redes Neuronales Hamiltonianas (HNN) clásicas fallan al concatenar y mezclar arbitrariamente variables de posición ($q$) y momento ($p$) dentro de las mismas capas ocultas densas [[1]](#referencias). Este entrelazamiento destruye la consistencia de las unidades físicas, degradando la geometría del sistema y promoviendo una acumulación catastrófica de errores numéricos a largo plazo [[1]](#referencias).

---

## 2. Experimentación Realizada
* Evaluación de la red bajo diferentes resoluciones de mallas temporales (Δt=0.01 y Δt=0.0005).
* Pruebas sistemáticas con múltiples funciones de activación pares frente a funciones impares (como tanh(x)) para verificar la estabilidad a largo plazo.
* Implementación de un esquema de normalización múltiple independiente a nivel de grado de libertad para mitigar el problema de Alto Rango Dinámico (HDR).

## 3. Flujograma de la Red (SLL-HNN)
* **Mapeo Lineal:** Proyección lineal independiente punto a punto de las posiciones y momentos físicos hacia el espacio latente.
* **Desacoplamiento:** Cálculo separado de las energías para formar el Hamiltoniano latente aditivo (H = T + V).
* **Predicción:** Extracción de gradientes simplécticos mediante diferenciación automática y escalado inverso para las predicciones dinámicas y cinemáticas.

## 4. Descripción del Mejor Modelo
* Arquitectura SLL-HNN compacta y optimizada a solo dos capas ocultas profundas de 256 neuronas.
* Subred de energía cinética (T) restringida a una formulación analítica paramétrica explícita, sin mezclar unidades.
* Subred de energía potencial (V) aproximada mediante un Perceptrón Multicapa (MLP) continuo y diferenciable.
* La red prescinde por completo de estabilizadores numéricos artificiales, forzando al modelo a aprender la dinámica inmaculadamente.

## 5. Predicciones del Modelo a Largo Plazo
* Estabilidad orbital macroscópica comprobada en una ventana de integración de 400 años.
* A pesar de variaciones orbitales sutiles inherentes, los cuerpos no escapan de sus trayectorias y mantienen un estado gravitacionalmente ligado.
* El sistema exhibe un nivel de ruido significativamente menor comparado con métodos previos y redes más profundas.

## 6. La Función de Activación Ganadora
* La función de activación óptima demostró ser la función par f(x) = log(1 + x * tanh(x)).
* El uso de esta función par (simétrica) previene el sobreajuste y la deriva numérica, alineándose con la física real donde la energía potencial depende de los términos pares de las distancias relativas.

## 7. Librerías Utilizadas
* [Insertar librería 1, e.g., PyTorch]
* [Insertar librería 2, e.g., NumPy]
* [Insertar librería 3, e.g., Matplotlib]

## 8. Acerca de
* **Autores:** Jae Hoon Lee y Jesus Martin Bautista Ramirez.
* **Afiliaciones:** Open Stack Inc. / Universidad Peruana de Ciencias Aplicadas (UPC).
* **DOI / Paper:** [Insertar enlace al paper]

### <a name="referencias"></a> Referencias

**[1]** Lee, J. H., & Bautista Ramirez, J. M. (En desarrollo). *Hamiltonian Neural Networks for Solar System*. Puedes consultar el código fuente, la experimentación completa y las actualizaciones de esta investigación en nuestro [Repositorio Oficial de GitHub](https://github.com/jae-hoon-daniel-lee/hamiltonian-neural-networks-for-solar-system).