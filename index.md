---
title: Inicio
layout: home
nav_order: 1
---

# Laboratorio de Vehículos Autónomos: Percepción Visual en QCar
{: .fs-9 }

Este repositorio documenta la implementación y análisis de sistemas de percepción visual y toma de decisiones en tiempo real utilizando la plataforma robótica **Quanser QCar**. El proyecto explora desde técnicas de visión por computadora clásica hasta modelos de aprendizaje profundo (Deep Learning) para navegación autónoma segura.

---

## Estado del Proyecto
- **Desarrolladores:** Adrían Bazaldua, Sebastián Enguilo y Haili Ávila
- **Institución:** Universidad Iberoamericana
- **Asignatura:** Vehículos Autónomos (Docente: Juan Camilo Tejada Orjuela)
- **Fecha:** Primavera 2026
- **Tecnologías clave:** Python, OpenCV, PyTorch (YOLO), Linux (NVIDIA Jetson TX2), Quanser APIs.

---

## Arquitectura General del Sistema

El desarrollo de esta práctica se basa en el paradigma fundamental de la robótica autónoma: el ciclo de **Percepción, Decisión y Acción**.

1. **Percepción:** Adquisición de datos del entorno mediante la cámara CSI frontal del QCar a 30 FPS.
2. **Procesamiento:** Interpretación de los datos visuales. Se divide en dos enfoques comparativos:
   - *Visión Clásica:* Segmentación en espacio HSV y filtrado morfológico.
   - *Modelo Inteligente:* Inferencia en tiempo real utilizando redes neuronales convolucionales (YOLOv5n/YOLOv8n).
3. **Decisión:** Máquina de estados jerárquica para priorizar maniobras (ej. Freno de emergencia vs. Evasión direccional).
4. **Acción:** Transmisión de señales de control (Throttle y Steering) mediante PWM a los actuadores físicos del vehículo a través de la tarjeta HIL.

---

## Organización de la Documentación

Navega por el menú lateral para explorar cada etapa del laboratorio:

* **[Introducción y Metodología](...):** Requisitos, hardware y configuración inicial.
* **[Visión Clásica y Control](...):** Desarrollo del algoritmo HSV, calibración y máquina de estados.
* **[Deep Learning con YOLO](...):** Implementación del modelo inteligente y análisis de inferencia.
* **[Análisis y Discusión](...):** Validación experimental, resolución de preguntas y conclusiones generales.