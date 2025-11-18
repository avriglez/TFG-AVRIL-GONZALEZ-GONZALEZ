#Caracterización de Haces Láser Mediante Inteligencia Artificial

#Trabajo de Fin de Grado (TFG)

Este proyecto explora la aplicación de Redes Neuronales Convolucionales (CNN) para la caracterización automática de haces láser. El objetivo principal es descomponer un perfil de intensidad de luz láser desconocido en su composición de modos Hermite-Gauss (HG), estimando no solo los modos presentes sino también sus pesos relativos.

El trabajo se enmarca en la óptica paraxial y la multiplexación modal, una técnica de vanguardia para incrementar la capacidad de los canales de comunicación óptica en espacio libre.

---

## Puntos Destacados y Resultados

Se implementaron y entrenaron modelos de Deep Learning sobre una extensa base de datos sintética generada numéricamente en Python, simulando superposiciones de modos HG de bajo orden (l, m <= 5).

| Caso de Estudio | Precisión de Modos (Accuracy) | Error Absoluto Medio (MAE) Global | MAE de Pesos |
| :--- | :--- | :--- | :--- |
| **2 Modos Simples** | **98.76%** | 0.0245 | 0.1466 |
| **2 Modos Desfasados** | 98.35% | 0.0354 | 0.1802 |
| **2 Modos c/ Ruido (sigma=0.20)**| 97.06% | 0.0559 | 0.2124 |
| **3 Modos Simples** | 96.52% | 0.0811 | 0.1987 |

##Conclusiones Clave
* El modelo propuesto demuestra una **alta robustez y precisión** (superiores al 96% en todos los casos) en la identificación modal, incluso bajo condiciones adversas como el desfase relativo y la presencia de ruido gaussiano.
* Los resultados validan la viabilidad de utilizar CNNs como herramientas eficientes para el **monitoreo y alineamiento** de sistemas ópticos, y abren la puerta a su aplicación en sistemas avanzados de comunicación óptica.

---

##Estructura del Proyecto

El repositorio está organizado en dos componentes principales: la **Simulación** y la **Red Neuronal**.

1. Simulación y Generación de Datos
El dataset sintético se generó en Python a partir de la expresión teórica de la intensidad de un haz Hermite-Gauss (Ec. 38/46), simulando las siguientes condiciones:
* **Modos HG:** Desde el (0,0) hasta el (5,5).
* **Superposición de dos modos:**
    * Simple (sin desfase).
    * Con desfase relativo (phi en [0, pi]).
    * Con diferentes niveles de ruido gaussiano (sigma en [0.05, 0.20]).
* **Superposición de tres modos:** Simple (sin desfase ni ruido).
* **Archivos relevantes:**
    * `SIMULACIÓN 2 MODOS SIMPLES.py`
    * `SIMULACIÓN 2 MODOS DESFASADOS.py`
    * `SIMULACIÓN 2 MODOS CON RUIDO.py`
    * `SIMULACIÓN 3 MODOS.py`

2. Modelos de Red Neuronal (CNN)
Se implementaron dos arquitecturas convolucionales utilizando Keras/TensorFlow, diseñadas para clasificar los modos (36 posibles) y realizar la regresión de sus pesos relativos.

* **`RED NEURONAL 2 MODOS.py`**: Implementa la red para la descomposición de dos modos, incluyendo una **función de pérdida personalizada** que gestiona la simetría de los modos y penaliza predicciones inconsistentes.
    * **Arquitectura:** 2x (Conv2D + MaxPooling2D) -> Flatten -> Dense(256) -> Dropout(0.3) -> Dense(38, sigmoid).
* **`RED NEURONAL 3 MODOS.py`**: Versión extendida para manejar tres modos.
    * **Arquitectura:** 3 bloques convolucionales -> 2x (Dense(128) + Dropout(0.4)) -> Salidas bifurcadas (Modos: Sigmoid, Pesos: Softmax) -> Concatenate.

---

##Metodología y Entorno

* **Autor:** Avril González González
* **Tutores:** Enrique Conejero Jarque y Francisco Javier Serrano Rodríguez
* **Institución:** Universidad de Salamanca, Facultad de Ciencias
* **Curso Académico:** 2024-2025

##Tecnologías
* **Lenguaje:** Python
* **Deep Learning:** Keras / TensorFlow
* **Cálculo Numérico y Óptico:** NumPy, SciPy
* **Visualización:** Matplotlib

---

##Perspectivas Futuras

* **Extensión Modal:** Ampliar la capacidad del modelo a modos Hermite-Gauss de órdenes superiores y a otros conjuntos de modos, como los **Laguerre-Gauss**.
* **Validación Experimental:** Contrastar los resultados de simulación con datos obtenidos de montajes experimentales reales.
* **Casos más complejos:** Explorar la caracterización de haces bajo condiciones de **turbulencia atmosférica** o con modos rotados.