---
title: Visión Clásica y Control
layout: default
nav_order: 4
---

# Visión Clásica y Control
{: .fs-9 }

Desarrollo del algoritmo HSV, calibración y máquina de estados para la evasión autónoma de obstáculos.
{: .fs-6 .fw-300 }

---

El sistema de evasión del QCar se construyó en dos etapas fundamentales: la identificación dinámica de los colores en el entorno del laboratorio y la ejecución de maniobras evasivas estructuradas.

## 1. Calibración del Espacio de Color (HSV)

En entornos reales, la iluminación cambia drásticamente. Programar colores fijos (hardcoding) en el espacio RGB suele causar fallos en la detección de objetos. Para solucionar esto, utilizamos el espacio de color **HSV (Hue, Saturation, Value)** y desarrollamos un script interactivo para capturar los valores dinámicos directamente desde la cámara del vehículo.

El algoritmo genera una Región de Interés (ROI) en el centro de la pantalla. Al presionar la barra espaciadora, escanea los píxeles dentro del recuadro, calcula los mínimos y máximos, y genera las matrices de tolerancia.

![Herramienta de Calibración HSV]({{ "/assets/img/calibrar_HSV.png" | relative_url }})
*Vista de la cámara del QCar ejecutando el recuadro de calibración (ROI).*

```python
import time
import cv2
import numpy as np
from Quanser.pal.utilities.vision import Camera2D

def main():
    sampleRate = 30.0
    sampleTime = 1.0 / sampleRate

    imageWidth = 640
    imageHeight = 480
    cameraId = "3"

    cam = Camera2D(cameraId=cameraId, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)

    # --- DICCIONARIO MAESTRO DE COLORES ---
    # Aquí viven tus calibraciones. Cada color tiene sus umbrales y el color BGR para dibujar la caja
    colores = {
        "amarillo": {
            "lower": np.array([6, 193, 81], dtype=np.uint8),
            "upper": np.array([36, 255, 201], dtype=np.uint8),
            "bgr": (0, 255, 255)  # Amarillo en BGR
        },
        
        "azul": {
            "lower": np.array([91, 193, 20], dtype=np.uint8),
            "upper": np.array([122, 255, 140], dtype=np.uint8),
            "bgr": (255, 0, 0)    # Azul en BGR
        }
    }

    print("--- Iniciando Detección Multicolor Simultánea ---")
    print("Buscando: Gorra Amarilla, Caja Morada y Termo Azul.")
    print("Presiona 'q' en la ventana de video para salir.\n")

    # Pre-calculamos el kernel para no hacerlo en cada frame y ahorrar CPU
    kernel = np.ones((5, 5), np.uint8)

    try:
        while True:
            t0 = time.time()
            ok = cam.read()
            if ok:
                frame = cam.imageData.copy()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # --- EL BARRIDO MULTICOLOR ---
                # Este ciclo revisa cada color de nuestro diccionario en el mismo frame
                for nombre, datos in colores.items():
                    
                    # 1. Creamos la máscara solo para este color
                    mask = cv2.inRange(hsv, datos["lower"], datos["upper"])
                    
                    # 2. Limpieza
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    # 3. Buscar contornos (Con la sintaxis correcta para OpenCV 3)
                    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # 4. Encontrar el objeto de este color
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 500:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cx = x + w // 2
                            cy = y + h // 2

                            # Dibujamos usando el color asignado en el diccionario
                            color_caja = datos["bgr"]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color_caja, 2)
                            cv2.circle(frame, (cx, cy), 5, color_caja, -1)
                            
                            # Etiqueta con el nombre del color
                            texto = f"{nombre.upper()} ({cx}, {cy})"
                            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja, 2)
                            
                            # Detenemos la búsqueda de contornos de ESTE color para ir al siguiente
                            break 

                # Solo mostramos la ventana principal final con todas las cajas dibujadas
                cv2.imshow(f"Deteccion Multicolor CSI {cameraId}", frame)

            # Salir con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Pacing
            elapsed = time.time() - t0
            sleepTime = sampleTime - (elapsed % sampleTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    except KeyboardInterrupt:
        pass
    finally:
        cam.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
## 2. Navegación y Máquina de Estados: Evasión Direccional

Una vez obtenidos los umbrales de color, se implementó el cerebro de navegación. Para evitar respuestas erráticas ("jaloneos") causadas por la pérdida momentánea de visión en un fotograma, el control se estructuró mediante una **Máquina de Estados Temporizada**.

A diferencia de un control reactivo simple, esta máquina bloquea el estado de evasión durante 1 segundo completo para garantizar que la maniobra se ejecute correctamente, venciendo la fricción estática y dinámica de las llantas al girar.

Las reglas de decisión son las siguientes:
1. **Maniobra en curso:** Si una evasión ya inició, el sistema ignora nuevas detecciones y mantiene la curva hasta cumplir 1 segundo exacto.
2. **Evasión Derecha (Azul):** Al detectar el marcador azul, el vehículo gira agresivamente a la derecha (`Steer = -0.45`) con un incremento de potencia (`Throttle = 0.10`) para compensar la fricción de la curva.
3. **Evasión Izquierda (Amarillo):** Al detectar el marcador amarillo, el vehículo gira a la izquierda (`Steer = +0.45`) con la misma compensación de potencia.
4. **Standby:** Si no hay objetos en la escena y no hay ninguna maniobra activa, el vehículo permanece completamente estático por seguridad.

![Detección de la gorra amarilla]({{ "/assets/img/det_gorra_bonita.png" | relative_url }})
*La máquina de estados detecta el marcador y ejecuta la evasión direccional correspondiente.*

### Código: `autonomo_vision.py`

```python
import time
import cv2
import numpy as np
from Quanser.pal.utilities.vision import Camera2D
from Quanser.product_QCar import QCar

def main():
    # --- CONFIGURACIÓN DE CÁMARA ---
    sampleRate = 30.0
    sampleTime = 1.0 / sampleRate
    imageWidth = 640
    imageHeight = 480
    cameraId = "3"
    cam = Camera2D(cameraId=cameraId, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)

    # --- CONFIGURACIÓN DE HARDWARE QCAR ---
    print("Iniciando conexión con hardware QCar...")
    try:
        car = QCar()
    except Exception as e:
        print("Error al abrir el QCar:", e)
        return

    mtr_cmd = np.array([0.0, 0.0], dtype=np.float64)
    leds = np.zeros(8, dtype=np.float64)

    # --- DICCIONARIO DE COLORES ---
    colores = {
        "azul": {
            "lower": np.array([91, 193, 20], dtype=np.uint8),
            "upper": np.array([122, 255, 140], dtype=np.uint8),
            "bgr": (255, 0, 0)
        },
        "amarillo": {
            "lower": np.array([6, 193, 81], dtype=np.uint8),
            "upper": np.array([36, 255, 201], dtype=np.uint8),
            "bgr": (0, 255, 255)
        }
    }

    kernel = np.ones((5, 5), np.uint8)

    print("\n--- Sistema de Navegación Autónoma: Evasión Direccional ---")
    print(" - Azul: Evasión pronunciada a la DERECHA.")
    print(" - Amarillo: Evasión pronunciada a la IZQUIERDA.")
    print("Presiona 'q' en la ventana de video o Ctrl+C en terminal para detener.\n")

    # --- VARIABLES DE ESTADO Y TEMPORIZADOR ---
    maniobra_activa = False
    tiempo_fin_maniobra = 0.0
    steer_maniobra = 0.0
    throttle_maniobra = 0.0
    estado_actual = "ESPERANDO OBJETIVO"
    color_texto = (0, 255, 0)

    try:
        while True:
            t0 = time.time()
            t_actual = time.time()  # Reloj maestro para la máquina de estados

            ok = cam.read()
            if ok:
                frame = cam.imageData.copy()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                encontro_azul = False
                encontro_amarillo = False

                # --- 1. BARRIDO DE VISIÓN ---
                for nombre, datos in colores.items():
                    mask = cv2.inRange(hsv, datos["lower"], datos["upper"])
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 800:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cx = x + w // 2
                            cy = y + h // 2

                            cv2.rectangle(frame, (x, y), (x + w, y + h), datos["bgr"], 2)
                            cv2.circle(frame, (cx, cy), 5, datos["bgr"], -1)
                            
                            if nombre == "azul":
                                encontro_azul = True
                            elif nombre == "amarillo":
                                encontro_amarillo = True
                            break

                # --- 2. EL CEREBRO: MÁQUINA DE ESTADOS DE EVASIÓN ---
                
                # PRIORIDAD 1: Continuar una evasión ya empezada
                if maniobra_activa:
                    if t_actual < tiempo_fin_maniobra:
                        # Seguimos ejecutando el giro cerrado
                        throttle = throttle_maniobra
                        steer = steer_maniobra
                    else:
                        # Se acabó el segundo de la maniobra, frenamos y esperamos
                        maniobra_activa = False
                        throttle = 0.0
                        steer = 0.0
                        estado_actual = "ESPERANDO OBJETIVO"
                        color_texto = (0, 255, 0)

                # PRIORIDAD 2: Nueva evasión por AZUL (Va hacia la Derecha)
                elif encontro_azul:
                    maniobra_activa = True
                    tiempo_fin_maniobra = t_actual + 1.0  # Duración: 1 segundo
                    throttle_maniobra = 0.10  # Más fuerza para la curva
                    steer_maniobra = -0.45    # (-) DERECHA: Giro máximo
                    
                    throttle = throttle_maniobra
                    steer = steer_maniobra
                    estado_actual = "AZUL: ESQUIVANDO DERECHA"
                    color_texto = (255, 0, 0)

                # PRIORIDAD 3: Nueva evasión por AMARILLO (Va hacia la Izquierda)
                elif encontro_amarillo:
                    maniobra_activa = True
                    tiempo_fin_maniobra = t_actual + 1.0  # Duración: 1 segundo
                    throttle_maniobra = 0.10  # Más fuerza para la curva
                    steer_maniobra = 0.45     # (+) IZQUIERDA: Giro máximo
                    
                    throttle = throttle_maniobra
                    steer = steer_maniobra
                    estado_actual = "AMARILLO: ESQUIVANDO IZQ"
                    color_texto = (0, 255, 255)

                # PRIORIDAD 4: Standby absoluto
                else:
                    throttle = 0.0
                    steer = 0.0
                    estado_actual = "ESPERANDO OBJETIVO"
                    color_texto = (0, 255, 0)

                # --- 3. ENVIAR COMANDOS AL HARDWARE ---
                mtr_cmd[0] = throttle
                mtr_cmd[1] = steer
                car.read_write_std(mtr_cmd, leds)

                cv2.putText(frame, estado_actual, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)
                cv2.imshow(f"Vision Autonoma QCar", frame)

                print(f"\rEstado: {estado_actual:<28} | Throttle: {throttle:.2f} | Steer: {steer:+.2f}   ", end="", flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\nCerrando programa...")
                break

            elapsed = time.time() - t0
            sleepTime = sampleTime - (elapsed % sampleTime)
            if sleepTime > 0:
                time.sleep(sleepTime)

    except KeyboardInterrupt:
        print("\n\nInterrupción por teclado.")
    finally:
        print("Deteniendo motores y cerrando cámara...")
        try:
            car.write_std(np.array([0.0, 0.0], dtype=np.float64), leds)
            car.terminate()
        except Exception:
            pass
        cam.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
## 3. Demostración Experimental en Video

A continuación se presenta el registro en video de las pruebas físicas en el laboratorio, donde se observa al QCar reaccionando en tiempo real a los estímulos visuales (Amarillo = Evasión Izquierda, Azul = Evasión Derecha).

<div style="text-align:center;">
  <iframe 
    width="700" 
    height="400" 
    src="https://www.youtube.com/embed/9yZMlZbk61E"
    title="Pruebas de evasión autónoma QCar"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>