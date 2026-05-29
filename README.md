# SLL-HNN: Separable Latent Linear Hamiltonian Neural Networks para el Sistema Solar

## 1. Motivación

Este proyecto nace de una profunda curiosidad personal por comprender la física del universo y mi fascinación por el desarrollo de modelos de *Machine Learning* informados por la física. Explorar el cosmos a través del código siempre ha sido un objetivo para mí, pero al abordar el problema gravitacional de N-cuerpos en el Sistema Solar, me encontré con una brecha crítica en la mecánica computacional actual [[1]](#referencias).

Aunque el aprendizaje profundo ha revolucionado la aproximación de funciones, descubrimos que los modelos convencionales ignoran el rigor matemático que exigen los sistemas celestes [[1]](#referencias). Por ello, el desarrollo de la arquitectura **SLL-HNN** está impulsado por dos necesidades fundamentales:

* **Preservación Estricta de Invariantes Físicos:** La necesidad imperativa de respetar la coherencia dimensional y preservar la estructura simpléctica para evitar que el sistema disipe energía artificialmente durante simulaciones a escalas de tiempo astronómicas [[1]](#referencias).
* **Superación de Límites Clásicos (El Problema del Mezclado):** Las Redes Neuronales Hamiltonianas (HNN) clásicas fallan al concatenar y mezclar arbitrariamente variables de posición ($q$) y momento ($p$) dentro de las mismas capas ocultas densas. Este entrelazamiento destruye la consistencia de las unidades físicas, degradando la geometría del sistema y promoviendo una acumulación catastrófica de errores numéricos a largo plazo [[1]](#referencias).

---

## 2. Experimentación Realizada

Este repositorio documenta de manera exclusiva el desarrollo, experimentación y validación de la arquitectura Separable Latent Linear Hamiltonian Neural Network (SLL-HNN). Este modelo constituye mi aporte individual a la investigación conjunta. En el artículo científico resultante, esta arquitectura se contrasta con el desarrollo paralelo propuesto por mi coautor para determinar el modelo global óptimo.

Para evaluar la estabilidad a largo plazo y la robustez de la red frente a problemas de Alto Rango Dinámico (HDR), se implementó una estrategia de *time-splitting* sobre un conjunto de datos total de 4000 años: 3200 años para entrenamiento y 800 años para *testing* y evaluación. 

A lo largo de los cuatro bloques de experimentación, los componentes base de la arquitectura (desacoplamiento estricto $T-V$, proyecciones de Hadamard en el espacio latente y la política estricta de no utilizar estabilizadores numéricos artificiales) se mantuvieron inalterables. Las variaciones se aplicaron únicamente sobre la topología interna y las funciones de activación:

### Fases del Proceso Experimental

#### Experimentación 1: Exploración Base (Malla Fina, $\Delta t = 0.0005$)
* **Arquitectura:** 3 capas ocultas profundas de 256 neuronas cada una, con salida escalar para la energía potencial ($V$).
* **Datos:** Alta densidad de información temporal.
* **Funciones de Activación Evaluadas:** Exploración amplia utilizando $\log(\cosh(x))$, $\log(1+x\tanh(x))$, $x\tanh(x)$, $\sqrt{x^2+1}-1$, y $\tanh(x)$.
* **Propósito:** Evaluar el impacto inicial de la paridad y la simetría geométrica sobre un volumen masivo de datos sin reducción temporal.

#### Experimentación 2: Ajuste de Capacidad (Malla Gruesa, $\Delta t = 0.01$)
* **Arquitectura:** 3 capas ocultas profundas, con el ancho ajustado a 240 neuronas por capa.
* **Datos:** Reducción de datos en un factor de 20 mediante una discretización temporal más gruesa.
* **Funciones de Activación Evaluadas:** Reducidas a las tres funciones más prometedoras: $\log(\cosh(x))$, $\log(1+x\tanh(x))$ y $\tanh(x)$.
* **Propósito:** Analizar la sensibilidad del modelo y la acumulación de errores de truncamiento local ante una drástica reducción en la densidad de información temporal.

#### Experimentación 3: Versión "Lite" (Malla Gruesa, $\Delta t = 0.01$)
* **Arquitectura:** 3 capas ocultas profundas, con el ancho compactado a 128 neuronas por capa.
* **Datos:** Reducción de datos en un factor de 20.
* **Funciones de Activación Evaluadas:** Filtradas exclusivamente a las dos funciones pares de mayor rendimiento: $\log(\cosh(x))$ y $\log(1+x\tanh(x))$.
* **Propósito:** Comprobar la viabilidad de un modelo con un menor número de parámetros por capa para retener la física del sistema sin perder el confinamiento orbital.

#### Experimentación 4: Versión "Lite V2" (Malla Gruesa, $\Delta t = 0.01$) - Arquitectura Óptima
* **Arquitectura:** Reducción estructural a 2 capas ocultas profundas, optimizadas con 256 neuronas cada una.
* **Datos:** Reducción de datos en un factor de 20.
* **Funciones de Activación Evaluadas:** Evaluación final entre las funciones de paridad par: $\log(\cosh(x))$ y $\log(1+x\tanh(x))$.
* **Resultado:** Esta configuración se consolidó como la arquitectura definitiva para la contribución SLL-HNN al artículo científico. Demostró empíricamente que la restricción en la profundidad de la red (2 capas), compensada con un ancho adecuado, suprime de manera óptima el sobreajuste y la deriva numérica a lo largo de los siglos.

---

**Nota sobre la Consistencia de los Registros:** 
Mientras que la publicación científica se enfoca en reportar los invariantes energéticos y gráficos para ventanas de integración de 400 años (específicamente de las experimentaciones 1 y 4), este repositorio contiene los códigos fuente y los registros de estabilidad completos correspondientes a los 800 años de *testing* para los cuatro ensayos experimentales descritos.

## 3. Flujograma de la Red (SLL-HNN)

<div align="center">

![Animación del flujo de la Arquitectura SLL-HNN](media%20SLL-HNN/SLLHNNFlowchartAnimation.gif)

*Animación del flujo físico y matemático a través de la arquitectura SLL-HNN.*

</div>

El procesamiento de la arquitectura garantiza un desacoplamiento físico y dimensional estricto, estructurado en las siguientes etapas fundamentales:

* **Normalización Vectorial:** Los vectores de entrada físicos correspondientes a las posiciones ($\vec{q}$) y momentos ($\vec{p}$) se normalizan inicialmente de acuerdo con el mallado de los datos, dividiéndose por sus respectivos tensores de escala ($S_q$ y $S_p$).
* **Mapeo Lineal (Transformador de Fase):** Proyección lineal independiente punto a punto hacia el espacio latente. Cada variable normalizada atraviesa un transformador de fase lineal ($\odot W + b$) para permitir una calibración paramétrica autónoma por cada grado de libertad.
* **Desacoplamiento Energético:** Cálculo totalmente separado de las energías fundamentales para evitar el entrelazamiento de unidades:
  * *Energía Potencial ($ V $):* El flujo de posición latente ($\vec{\tilde{q}}$) ingresa a un Perceptrón Multicapa (MLP) continuo, el cual comprime la información y escupe finalmente un valor escalar representativo del potencial gravitacional.
  * *Energía Cinética ($ T $):* El flujo de momento latente ($\vec{\tilde{p}}$) interactúa con una formulación analítica paramétrica a través de una matriz de masa inversa aprendida ($M^{-1}$), resultando en el escalar de energía cinética.
* **Fusión Hamiltoniana:** Ambos escalares convergen y se suman coherentemente para formar el Hamiltoniano latente aditivo ($H = T + V$).
* **Predicción (Autograd Simpléctico):** Extracción de gradientes simplécticos mediante diferenciación automática. El sistema deriva el Hamiltoniano respecto a los estados latentes ($\dot{{\vec{r}}}\_{\text{lat}} = \frac{dH}{d\vec{p}}$ y $\dot{{\vec{p}}}\_{\text{lat}} = -\frac{dH}{d\vec{r}}$).
* **Escalado Inverso (Denormalización):** Las derivadas latentes resultantes pasan por una transformación lineal puramente multiplicativa (estrictamente sin matriz de sesgo) y son multiplicadas por sus tensores de escala de salida ($S_{\dot{r}}$, $S_{\dot{p}}$) para entregar las predicciones dinámicas y cinemáticas físicas finales en su magnitud real.

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