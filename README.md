# SLL-HNN: Separable Latent Linear Hamiltonian Neural Networks para el Sistema Solar

## 1. Motivación
* Necesidad de preservar invariantes físicos y la coherencia dimensional en la simulación del sistema gravitacional de N-cuerpos.
* Limitaciones de las HNN clásicas al mezclar variables de posición y momento en las mismas capas, lo que degrada la geometría simpléctica y acumula errores numéricos.

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