---
title: Introducción y Metodología
layout: default
nav_order: 2
---

# Introducción y Metodología
{: .fs-9 }

Marco teórico, objetivos y metodología experimental aplicada para lograr la navegación autónoma básica del Quanser QCar mediante percepción visual.
{: .fs-6 .fw-300 }

---

## Objetivos del Laboratorio

El objetivo principal de esta práctica es implementar una prueba experimental de visión por computador en el QCar, empleando su cámara frontal, con el fin de detectar un estímulo visual en tiempo real y asociarlo con una acción de navegación segura.

### Objetivos Específicos

1. Configurar la adquisición de imágenes desde la cámara frontal CSI utilizando la API oficial de Quanser.
2. Aplicar operaciones de preprocesamiento y segmentación de imágenes con OpenCV.
3. Detectar visualmente un objeto o marcador de color en tiempo real.
4. Diseñar una lógica de decisión para generar una respuesta de navegación y ejecutar un control seguro.

---

## Plataforma Experimental: Quanser QCar

El desarrollo se llevó a cabo sobre la plataforma Quanser QCar, un vehículo a escala diseñado para la investigación y desarrollo en conducción autónoma.

### Especificaciones de Hardware y Software

- **Cámara Frontal:** Se utilizó la cámara frontal CSI, identificada en el sistema de captura como el dispositivo `3` (`video://localhost:3`).
- **Procesamiento Embebido:** NVIDIA Jetson ejecutando un entorno Linux Ubuntu.
- **Lenguaje y Librerías:** Python 3. Se emplearon las librerías `numpy` y `opencv-python` para el procesamiento de imágenes, junto con las APIs propietarias `quanser.multimedia` (para el flujo de video) y `quanser.hardware` (para la comunicación de bajo nivel con los motores).

---

## Metodología: El Ciclo de Percepción y Acción

El diseño del sistema autónomo se basa en un flujo continuo de tres etapas fundamentales, ejecutadas en bucle cerrado a 30 fotogramas por segundo:

1. **Percepción (Captura y Preprocesamiento):** La cámara captura el entorno en tiempo real. La imagen original se convierte para aislar los objetos de interés, descartando información innecesaria del fondo.
2. **Decisión (Extracción de Características):** Mediante algoritmos de visión artificial, se determina la presencia, tamaño y ubicación exacta (centroide) de los marcadores en la pantalla.
3. **Acción (Control de Bajo Nivel):** Dependiendo de la ubicación del objeto, el "cerebro" del vehículo calcula las señales de control necesarias. Estas señales se envían a la tarjeta HIL (Hardware-In-the-Loop) escribiendo directamente en los canales de PWM (acelerador) y dirección.

> **Protocolo de Seguridad:** Para garantizar la integridad del equipo, el desarrollo de la lógica de decisión se realizó inicialmente con el vehículo elevado, validando la rotación de los ejes sin contacto con el suelo. Adicionalmente, se programaron rutinas de interrupción (`finally`) para asegurar el apagado total de los motores (`PWM = 0.0`) ante cualquier excepción en el código.