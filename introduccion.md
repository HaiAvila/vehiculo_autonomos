---
title: Introducción y Metodología
layout: default
nav_order: 2
---

# Introducción y Metodología
{: .fs-9 }

Esta sección describe el marco teórico, los objetivos y la metodología experimental aplicada para lograr la navegación autónoma básica del Quanser QCar mediante percepción visual.

---

## Objetivos del Laboratorio

[cite_start]El objetivo principal de esta práctica es implementar una prueba experimental de visión por computador en el QCar, empleando su cámara frontal, con el fin de detectar un estímulo visual en tiempo real y asociarlo con una acción de navegación segura[cite: 16].

### Objetivos Específicos
1. [cite_start]Configurar la adquisición de imágenes desde la cámara frontal CSI utilizando la API oficial de Quanser[cite: 18].
2. [cite_start]Aplicar operaciones de preprocesamiento y segmentación de imágenes con OpenCV[cite: 19].
3. [cite_start]Detectar visualmente un objeto o marcador de color en tiempo real[cite: 20].
4. [cite_start]Diseñar una lógica de decisión para generar una respuesta de navegación y ejecutar un control seguro[cite: 21, 22].

---

## Plataforma Experimental: Quanser QCar

El desarrollo se llevó a cabo sobre la plataforma Quanser QCar, un vehículo a escala diseñado para la investigación y desarrollo en conducción autónoma. 

### Especificaciones de Hardware y Software
* [cite_start]**Cámara Frontal:** Se utilizó la cámara frontal CSI, identificada en el sistema de captura como el dispositivo `3` (`video://localhost:3`)[cite: 41, 75].
* [cite_start]**Procesamiento Embebido:** NVIDIA Jetson ejecutando un entorno Linux Ubuntu[cite: 45].
* [cite_start]**Lenguaje y Librerías:** Python 3. Se emplearon las librerías `numpy` y `opencv-python` para el procesamiento de imágenes, junto con las APIs propietarias `quanser.multimedia` (para el flujo de video) y `quanser.hardware` (para la comunicación de bajo nivel con los motores)[cite: 46, 47].

---

## Metodología: El Ciclo de Percepción y Acción

[cite_start]El diseño del sistema autónomo se basa en un flujo continuo de tres etapas fundamentales, ejecutadas en bucle cerrado a 30 fotogramas por segundo[cite: 32]:

1. **Percepción (Captura y Preprocesamiento):** La cámara captura el entorno en tiempo real. [cite_start]La imagen original se convierte para aislar los objetos de interés, descartando información innecesaria del fondo[cite: 33].
2. [cite_start]**Decisión (Extracción de Características):** Mediante algoritmos de visión artificial, se determina la presencia, tamaño y ubicación exacta (centroide) de los marcadores en la pantalla[cite: 34].
3. **Acción (Control de Bajo Nivel):** Dependiendo de la ubicación del objeto, el "cerebro" del vehículo calcula las señales de control necesarias. [cite_start]Estas señales se envían a la tarjeta HIL (Hardware-In-the-Loop) escribiendo directamente en los canales de PWM (acelerador) y dirección[cite: 278].

> **Protocolo de Seguridad:** Para garantizar la integridad del equipo, el desarrollo de la lógica de decisión se realizó inicialmente con el vehículo elevado, validando la rotación de los ejes sin contacto con el suelo. [cite_start]Adicionalmente, se programaron rutinas de interrupción (`finally`) para asegurar el apagado total de los motores (`PWM = 0.0`) ante cualquier excepción en el código[cite: 61, 64].