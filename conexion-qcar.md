---
layout: default
title: Conexión y Entorno
nav_order: 3
---

# Configuración del Entorno y Conexión Remota
{: .fs-9 }

Arquitectura de trabajo remoto para programar y ejecutar los algoritmos de visión artificial directamente en la Jetson Nano del QCar desde una MacBook Pro.
{: .fs-6 .fw-300 }

---

Para el desarrollo de este proyecto se estableció una arquitectura de trabajo remoto que permitió editar y desplegar los algoritmos de visión artificial directamente en el sistema embebido del QCar, utilizando una MacBook Pro como estación de control. Esta configuración elimina la necesidad de periféricos físicos conectados al robot y centraliza el flujo de trabajo en un único entorno de desarrollo.

## 1. Instalación del Servidor Gráfico (XQuartz)

Dado que el QCar opera sobre un sistema Linux (Ubuntu) y los algoritmos de procesamiento de imagen generan ventanas de visualización mediante OpenCV, se requiere una solución para renderizar esas interfaces gráficas en macOS. Para ello se instaló **XQuartz**, una implementación del sistema de ventanas X11 para Mac.

XQuartz habilita el protocolo de **X11 Forwarding**, que permite que cualquier ventana generada en el proceso remoto (corriendo en la Jetson Nano) sea transmitida y desplegada localmente en la pantalla de la MacBook. Sin este componente, los llamados a `cv2.imshow()` fallarían al no encontrar un servidor gráfico disponible.

## 2. Conexión SSH con Redirección Gráfica

La comunicación con el QCar se realiza mediante el protocolo SSH. Para habilitar el reenvío de señales gráficas y visualizar la cámara en tiempo real desde la Mac, se utiliza la bandera `-X`, que activa X11 Forwarding sobre la sesión segura.

```bash
ssh -X nvidia@192.168.0.219
```

Este comando establece una terminal remota en la Jetson Nano del robot. Cualquier ventana generada por el código Python durante la sesión se renderiza automáticamente en el monitor local.

## 3. Control de Versiones y Repositorio

El código fuente del proyecto se aloja y versiona en el repositorio público de GitHub:

**[`Adr1anBaz/qcar_project`](https://github.com/Adr1anBaz/qcar_project)**

Este repositorio centraliza todos los scripts de percepción, control y configuración, y sirve como canal de despliegue hacia el hardware del robot.

### Inicialización en el QCar

Una vez conectado por SSH, se preparó el directorio de trabajo en la Jetson y se clonó el repositorio:

```bash
# 1. Crear el directorio de trabajo
mkdir -p ~/Desktop/qcar_project
cd ~/Desktop/qcar_project

# 2. Clonar el repositorio
git clone https://github.com/Adr1anBaz/qcar_project.git .
```

> **Nota sobre dependencias:** La instalación de paquetes específicos (como los requeridos para YOLOv8 y las APIs de comunicación de Quanser) se gestiona a través de un entorno virtual de Python. Esto mantiene el sistema base del QCar limpio y evita conflictos entre versiones de librerías. El procedimiento detallado se describe en la sección de **Deep Learning con YOLO**.

## 4. Desarrollo en Visual Studio Code

Se utilizó la extensión **Remote Repositories** de VS Code para conectarse directamente al repositorio de GitHub y trabajar sobre los archivos del proyecto sin necesidad de clonar localmente. Este flujo de trabajo integrado permitió:

- Editar scripts con las herramientas nativas de la Mac (linting, autocompletado, extensiones).
- Sincronizar cambios de forma inmediata mediante commits y pushes al repositorio remoto.
- Desplegar actualizaciones en el robot ejecutando `git pull` en la terminal SSH, cerrando el ciclo de desarrollo.
- Monitorear la telemetría visual del vehículo en tiempo real a través de la conexión X11.

Este enfoque redujo significativamente la fricción del ciclo de iteración, permitiendo pasar de una modificación en el código a una prueba en el hardware físico en cuestión de segundos.
