# SLL-HNN: Separable Latent Linear Hamiltonian Neural Networks para el Sistema Solar

## 1. Motivación

Este proyecto nace de una profunda curiosidad personal por comprender la física del universo y mi fascinación por el desarrollo de modelos de *Machine Learning* informados por la física. Explorar el cosmos a través del código siempre ha sido un objetivo para mí, pero al abordar el problema gravitacional de N-cuerpos en el Sistema Solar, me encontré con una brecha crítica en la mecánica computacional actual [[1]](#referencias).

Aunque el aprendizaje profundo ha revolucionado la aproximación de funciones, los modelos convencionales ignoran el rigor matemático que exigen los sistemas celestes [[1]](#referencias). Como una variante paralela a la arquitectura base Plausible HNN (P-HNN), el desarrollo de la arquitectura SLL-HNN está impulsado por dos necesidades fundamentales:

* **Preservación Estricta de Invariantes Físicos:** La necesidad imperativa de respetar la coherencia dimensional y preservar la estructura simpléctica para evitar que el sistema disipe energía artificialmente durante simulaciones a escalas de tiempo astronómicas.
* **Superación de Límites Clásicos (El Problema del Mezclado):** Las Redes Neuronales Hamiltonianas (HNN) clásicas fallan al concatenar y mezclar arbitrariamente variables de posición ($q$) y momento ($p$) dentro de las mismas capas ocultas densas. Este entrelazamiento destruye la consistencia de las unidades físicas, degradando la geometría del sistema y promoviendo una acumulación catastrófica de errores numéricos a largo plazo.

---

## 2. Experimentación Realizada

Este repositorio documenta de manera exclusiva el desarrollo, experimentación y validación de la arquitectura Separable Latent Linear Hamiltonian Neural Network (SLL-HNN). Este modelo constituye mi aporte individual a la investigación conjunta, sirviendo como una prueba de concepto independiente que demuestra que la bifurcación explícita de la energía logra la estabilidad macroscópica.

Para evaluar la estabilidad a largo plazo y la robustez de la red frente a problemas de Alto Rango Dinámico (HDR), se implementó una estrategia de time-splitting sobre un conjunto de datos total de 4000 años: 3200 años para entrenamiento y 800 años para testing y evaluación. 

A lo largo de los cuatro bloques de experimentación, los componentes base de la arquitectura (desacoplamiento estricto $T-V$, proyecciones de Hadamard en el espacio latente y la estricta política de no utilizar estabilizadores numéricos artificiales) se mantuvieron inalterables. Las variaciones se aplicaron únicamente sobre la topología interna y las funciones de activación:

### Fases del Proceso Experimental

#### Experimentación 1: Exploración Base (Malla Fina, $\Delta t=0.0005$)
* **Arquitectura:** 3 capas ocultas profundas de 256 neuronas cada una, con salida escalar para la energía potencial ($V$).
* **Datos:** Alta densidad de información temporal.
* **Funciones de Activación Evaluadas:** Exploración amplia utilizando $\log(\cosh(x))$, $\log(1+x\tanh(x))$, $x\tanh(x)$, $\sqrt{x^2+1}-1$, y $\tanh(x)$.
* **Propósito:** Evaluar el impacto inicial de la paridad y la simetría geométrica sobre un volumen masivo de datos sin reducción temporal.

#### Experimentación 2: Ajuste de Capacidad (Malla Gruesa, $\Delta t=0.01$)
* **Arquitectura:** 3 capas ocultas profundas, con el ancho ajustado a 240 neuronas por capa.
* **Datos:** Reducción de datos en un factor de 20 mediante una discretización temporal más gruesa.
* **Funciones de Activación Evaluadas:** Reducidas a las tres funciones más prometedoras: $\log(\cosh(x))$, $\log(1+x\tanh(x))$ y $\tanh(x)$.
* **Propósito:** Analizar la sensibilidad del modelo y la acumulación de errores de truncamiento local ante una drástica reducción en la densidad de información temporal.

#### Experimentación 3: Versión "Lite" (Malla Gruesa, $\Delta t=0.01$)
* **Arquitectura:** 3 capas ocultas profundas, con el ancho compactado a 128 neuronas por capa.
* **Datos:** Reducción de datos en un factor de 20.
* **Funciones de Activación Evaluadas:** Filtradas exclusivamente a las dos funciones pares de mayor rendimiento: $\log(\cosh(x))$ y $\log(1+x\tanh(x))$.
* **Propósito:** Comprobar la viabilidad de un modelo con un menor número de parámetros por capa para retener la física del sistema sin perder el confinamiento orbital.

#### Experimentación 4: Versión "Lite V2" (Malla Gruesa, $\Delta t=0.01$) - Arquitectura Óptima
* **Arquitectura:** Reducción estructural a 2 capas ocultas profundas, optimizadas con 256 neuronas cada una.
* **Datos:** Reducción de datos en un factor de 20.
* **Funciones de Activación Evaluadas:** Evaluación final entre las funciones de paridad par.
* **Resultado:** Esta configuración se consolidó como la arquitectura definitiva para la contribución SLL-HNN al artículo científico. Demostró empíricamente que una topología más compacta evita las inestabilidades geométricas de redes más profundas, mitigando el sobreajuste y la deriva numérica a lo largo de los siglos.

> **Nota sobre la Consistencia de los Registros:** Mientras que la publicación científica final reporta el horizonte acotado de 600 años de estabilidad macroscópica (obtenido de la configuración óptima), este repositorio contiene los códigos fuente y los registros de estabilidad completos correspondientes a los 800 años totales de testing para todos los ensayos experimentales.

---

## 3. Flujograma e Internos de la Red (SLL-HNN)

<div align="center">
  <a href="ENLACE_COMPLETO_DE_TU_VIDEO_DE_YOUTUBE" target="_blank">
    <img src="https://img.youtube.com/vi/ID_DE_TU_VIDEO/maxresdefault.jpg" alt="Animación del flujo de la Arquitectura SLL-HNN">
  </a>
  <p><em>Animación del flujo físico y matemático a través de la arquitectura SLL-HNN. Haz clic en la imagen para ver el video.</em></p>
</div>

El procesamiento de la arquitectura garantiza un desacoplamiento físico y dimensional estricto, estructurado en las siguientes etapas fundamentales:

* **Normalización Vectorial:** Los vectores de entrada físicos correspondientes a las posiciones ($\vec{q}$) y momentos ($\vec{p}$) se normalizan inicialmente dividiéndose por sus respectivos tensores escalares de valores máximos absolutos ($\max(|q|)$ y $\max(|p|)$).
* **Mapeo Lineal (Transformador de Fase):** Proyección lineal independiente punto a punto hacia el espacio latente. Cada variable normalizada atraviesa un transformador de fase lineal ($\odot W+b$) para permitir una calibración paramétrica autónoma por cada grado de libertad.
* **Desacoplamiento Energético:** Cálculo totalmente separado de las energías fundamentales para evitar el entrelazamiento de unidades:
  * *Energía Potencial ($V$):* El flujo de posición latente ($\tilde{q}$) ingresa a un Perceptrón Multicapa (MLP) continuo, el cual comprime la información y escupe finalmente un valor escalar representativo del potencial gravitacional.
  * *Energía Cinética ($T$):* El flujo de momento latente ($\tilde{p}$) interactúa con una formulación analítica paramétrica a través de una matriz de masa inversa aprendida ($M^{-1}$), resultando en el escalar de energía cinética.
* **Fusión Hamiltoniana:** Ambos escalares convergen y se suman coherentemente para formar el Hamiltoniano latente aditivo ($H=T+V$).
* **Predicción (Autograd Simpléctico):** Extracción de gradientes simplécticos mediante diferenciación automática. El sistema deriva el Hamiltoniano respecto a los estados latentes ($\dot{\tilde{q}}=\frac{\partial H}{\partial\tilde{p}}$ y $\dot{\tilde{p}}=-\frac{\partial H}{\partial\tilde{q}}$).
* **Escalado Inverso (Denormalización):** Las derivadas latentes resultantes pasan por una transformación lineal puramente multiplicativa (estrictamente sin matriz de sesgo) y son multiplicadas por sus factores de escala de salida ($\max(|\dot{q}|)$ y $\max(|\dot{p}|)$) para entregar las predicciones dinámicas y cinemáticas físicas finales.

### Visualización del Entrenamiento y Pesos Lineales

A continuación se detalla el comportamiento interno de la red durante el proceso de optimización del mejor modelo (Experimentación 4):

<div align="center">
  <img src="media%20SLL-HNN/loss.png" alt="Curva de Pérdida del SLL-HNN">
  <p><em>Evolución de la función de pérdida durante 1000 épocas para la arquitectura óptima SLL-HNN de 2 capas.</em></p>

  <img src="media%20SLL-HNN/SLL_W_B.png" alt="Pesos Lineales y Sesgo del Encoder">
  <p><em>Distribución de la matriz de pesos lineales y sesgo (bias) en la fase de codificación (Encoder) hacia el espacio latente.</em></p>

  <img src="media%20SLL-HNN/SLL_W_dot.png" alt="Pesos Lineales del Decoder">
  <p><em>Matrices de pesos correspondientes a la decodificación de las derivadas temporales tras la diferenciación simpléctica.</em></p>
</div>

---

## 4. Descripción del Mejor Modelo
* **Topología Optimizada:** Reducción estructural a una configuración compacta de 2 capas ocultas profundas con un ancho de 256 neuronas cada una.
* **Huella de Memoria:** El modelo se condensa en un tamaño de 600 kB con un total exacto de 74,195 parámetros.
* **Bifurcación Explícita:** Subred de energía cinética ($T$) restringida a una formulación analítica paramétrica explícita, separada estrictamente de la subred de energía potencial ($V$) aproximada mediante el MLP continuo.
* **Física Limpia:** La red prescinde por completo de estabilizadores numéricos artificiales, forzando al modelo a aprender la dinámica inmaculadamente sin ensuciar los datos físicos.

## 5. Predicciones del Modelo a Largo Plazo
* Estabilidad orbital macroscópica comprobada y confinada dentro de un horizonte de integración de 600 años.
* A pesar de variaciones orbitales sutiles inherentes al mapeo puramente lineal, los cuerpos evitan eyecciones no físicas y mantienen un estado gravitacionalmente ligado.
* La configuración demuestra empíricamente que imponer invariantes físicos a través de activaciones pares mantiene la estabilidad, incluso bajo las restricciones representacionales de embeddings estrictamente lineales.

## 6. La Función de Activación Ganadora
* La función de activación óptima demostró ser la función compuesta par $f(x)=\log(1+x\tanh(x))$.
* El uso de esta función par previene el sobreajuste y mitiga la deriva numérica severa observada en redes tradicionales, alineándose con la simetría espacial de la física real, donde la energía potencial depende de términos pares de las distancias relativas.

## 7. Librerías Utilizadas
* PyTorch
* NumPy
* Matplotlib

## 8. Ecosistema de Repositorios de la Investigación

Para garantizar la reproducibilidad y modularidad de los resultados presentados en el artículo, el código y los datos de la investigación están distribuidos en tres repositorios específicos:

* **Arquitectura SLL-HNN (Este repositorio):**
  Contiene exclusivamente mi investigación paralela, los registros de estabilidad temporal y la implementación de la arquitectura de mapeo lineal latente.
* **Generación del Dataset (Integrador Simpléctico de Yoshida):**
  [https://github.com/Assia-Network/Yoshida-Symplectic-Integrator.git](https://github.com/Assia-Network/Yoshida-Symplectic-Integrator.git)
  Repositorio de mi autoría que alberga el motor numérico utilizado para generar los 4000 años de datos de entrenamiento y evaluación. Implementa un integrador de Yoshida de alto orden para asegurar que los datos base conserven la geometría simpléctica del Sistema Solar.
* **Arquitecturas P-HNN y BLK-P-HNN:**
  [https://github.com/jae-hoon-daniel-lee/hamiltonian-neural-networks-for-solar-system.git](https://github.com/jae-hoon-daniel-lee/hamiltonian-neural-networks-for-solar-system.git)
  Repositorio administrado por mi coautor, Jae Hoon Lee. Documenta la arquitectura base de la investigación (Plausible HNN) y los experimentos correspondientes a la red esparcida enfocada en la compresión de parámetros (Block Sparse P-HNN).

## 9. Acerca de
* **Autores:** Jae Hoon Lee y Jesus Martin Bautista Ramirez.
* **Afiliaciones:** Open Stack Inc. / Universidad Peruana de Ciencias Aplicadas (UPC).
* **DOI / Paper:** [Enlace pendiente de publicación]

### <a name="referencias"></a> Referencias

**[1]** Lee, J. H., & Bautista Ramirez, J. M. (En desarrollo). *Hamiltonian Neural Networks for Solar System*. Documentación, código fuente y desarrollo conjunto disponibles a través de los repositorios listados en la sección 8.