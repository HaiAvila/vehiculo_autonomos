---
layout: default
title: Navegación Autónoma y Arquitectura Distribuida
nav_order: 6
---

# Navegación Autónoma y Arquitectura Distribuida
{: .fs-9 }

Implementación de una arquitectura Cliente-Servidor por sockets para el procesamiento de redes neuronales y control del vehículo basado en una Máquina de Estados Finita (FSM).
{: .fs-6 .fw-300 }

---

El despliegue final del modelo YOLOv8 directamente sobre la NVIDIA Jetson Nano del QCar presentó restricciones técnicas inherentes al hardware de borde (*Edge Computing*). Las dependencias modernas de la librería `ultralytics` entraban en conflicto con la versión de Python estática requerida por las APIs de control físico de Quanser, generando un escenario de incompatibilidad conocido como *Dependency Hell*.

Para resolver esto sin comprometer la integridad del sistema base del vehículo, se diseñó una **Arquitectura Distribuida (Cliente-Servidor)** a través de la red local (Wi-Fi).

## 1. Arquitectura de Computación Distribuida

El sistema se dividió en dos nodos operativos principales, comunicados mediante un *socket* TCP/IP de alta velocidad:

- **Nodo Cliente (Los Ojos y Músculos — QCar):** Ejecuta un script ligero que captura fotogramas de la cámara CSI, los comprime en formato JPEG y los transmite por Wi-Fi. Simultáneamente, queda a la escucha de comandos de texto plano (ej. `"AVANZA"`, `"ALTO"`) para accionar los motores mediante la librería nativa de Quanser.
- **Nodo Servidor (El Cerebro — MacBook Pro):** Recibe el flujo de video en bruto, procesa la inferencia utilizando los pesos óptimos (`best.pt`) de YOLOv8, y ejecuta la lógica de toma de decisiones. Una vez determinado el siguiente estado del vehículo, envía la instrucción de regreso al QCar.

### Entorno Virtual en macOS

Dado que las versiones recientes de macOS bloquean la instalación global de paquetes mediante `pip` (política *externally-managed-environment*), el nodo servidor se configuró y ejecutó dentro de un **Entorno Virtual (venv)** aislado. Esto garantizó la ejecución limpia de OpenCV y Ultralytics sin comprometer la seguridad del sistema operativo anfitrión.

## 2. Máquina de Estados Finita (FSM) y Toma de Decisiones

Para la navegación autónoma dentro de la pista de 2.5m × 2.5m, se programó una toma de decisiones basada en reglas (*Rule-Based Decision Making*). Dado que el QCar utiliza dirección tipo Ackermann (no puede girar sobre su propio eje), las maniobras de giro a 90 grados y los cambios de carril se resolvieron aplicando el volante al máximo durante intervalos de tiempo precisos.

La FSM se estructuró con un sistema estricto de prioridades y **Tiempos de Inmunidad** para evitar bucles lógicos infinitos:

1. **Prioridad 0 (Freno de Emergencia):** La detección de la clase `persona` anula cualquier comando. Frena el vehículo instantáneamente. Además, implementa un *Time Freeze* (congelación de tiempo): si el carro estaba a mitad de una curva, los temporizadores se pausan, permitiendo que el vehículo retome y termine su maniobra exactamente donde se quedó una vez que el peatón despeja la vía.
2. **Prioridad 1 (Maniobras Cronometradas):** Si hay una maniobra activa (ej. un giro de 3 segundos), el vehículo ignora nuevas señales visuales hasta terminar el movimiento.
3. **Prioridad 2 (Evaluación con Inmunidad):** Para evitar que el vehículo reaccione múltiples veces a la misma señal mientras la rebasa, se asignó un reloj de "inmunidad". Tras evadir un cono o pasar un semáforo verde, el QCar ignora esa clase específica por 3 segundos, forzando un estado de crucero (`AVANZA`) en línea recta.

## 3. Código Fuente Final

### Nodo Servidor (`servidor_yolo.py`) — Ejecutado en macOS

```python
import cv2
import socket
import struct
import numpy as np
import time
from ultralytics import YOLO

def main():
    print("[+] Cargando red neuronal en la Mac...")
    model = YOLO("best.pt")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(1)
    print("\n[+] Mac escuchando en el puerto 9999. Esperando al QCar...")

    conn, addr = server_socket.accept()
    print(f"[!] ¡QCar conectado desde {addr}!")

    data = b""
    payload_size = struct.calcsize("<L")

    # --- MEMORIA Y TEMPORIZADORES DEL ROBOT ---
    estado_actual = "ALTO"
    texto_pantalla = "INICIANDO"

    maniobra_actual = None
    tiempo_fin_maniobra = 0.0
    tiempo_inmunidad_stop = 0.0
    tiempo_inmunidad_verde = 0.0
    tiempo_inmunidad_cono = 0.0

    try:
        while True:
            # --- 1. Recibir imagen del QCar ---
            while len(data) < payload_size:
                data += conn.recv(4096)

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("<L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            # --- 2. Procesar imagen con YOLO ---
            resultados = model(frame, verbose=False)
            clases_detectadas = []
            if len(resultados[0].boxes) > 0:
                ids = resultados[0].boxes.cls.tolist()
                clases_detectadas = [model.names[int(c)] for c in ids]

            t_actual = time.time()

            # --- 3. LÓGICA DE VEHÍCULO AUTÓNOMO (FSM) ---

            # PRIORIDAD 0: Freno de emergencia absoluto
            if "persona" in clases_detectadas:
                estado_actual = "ALTO"
                texto_pantalla = "PELIGRO: PEATON DETECTADO"

                # "Time Freeze": Se pausa el desgaste de las maniobras
                if maniobra_actual is not None:
                    tiempo_fin_maniobra += 0.1
                tiempo_inmunidad_stop += 0.1
                tiempo_inmunidad_verde += 0.1
                tiempo_inmunidad_cono += 0.1

            # PRIORIDAD 1: Ejecución de maniobra cronometrada
            elif t_actual < tiempo_fin_maniobra:
                tiempo_restante = tiempo_fin_maniobra - t_actual

                if maniobra_actual == "STOP_GIRO":
                    estado_actual = "GIRO_DER"
                    texto_pantalla = f"GIRO 90 DERECHA ({tiempo_restante:.1f}s)"
                    if tiempo_restante < 0.2:
                        tiempo_inmunidad_stop = t_actual + 3.0

                elif maniobra_actual == "VERDE_GIRO":
                    estado_actual = "GIRO_IZQ"
                    texto_pantalla = f"GIRO 90 IZQUIERDA ({tiempo_restante:.1f}s)"
                    if tiempo_restante < 0.2:
                        tiempo_inmunidad_verde = t_actual + 3.0

                elif maniobra_actual == "CONO":
                    estado_actual = "EVADE_DER"
                    texto_pantalla = f"EVADIENDO CONO ({tiempo_restante:.1f}s)"
                    if tiempo_restante < 0.2:
                        tiempo_inmunidad_cono = t_actual + 3.0

            # PRIORIDAD 2: Evaluación de señales nuevas
            else:
                maniobra_actual = None

                if "stop" in clases_detectadas and t_actual > tiempo_inmunidad_stop:
                    maniobra_actual = "STOP_GIRO"
                    tiempo_fin_maniobra = t_actual + 3.0
                    estado_actual = "GIRO_DER"
                    texto_pantalla = "INICIANDO GIRO DER"

                elif "semaforo_verde" in clases_detectadas and t_actual > tiempo_inmunidad_verde:
                    maniobra_actual = "VERDE_GIRO"
                    tiempo_fin_maniobra = t_actual + 3.0
                    estado_actual = "GIRO_IZQ"
                    texto_pantalla = "INICIANDO GIRO IZQ"

                elif "cono" in clases_detectadas and t_actual > tiempo_inmunidad_cono:
                    maniobra_actual = "CONO"
                    tiempo_fin_maniobra = t_actual + 2.0
                    estado_actual = "EVADE_DER"
                    texto_pantalla = "CAMBIO DE CARRIL 2s"

                # Estado de inmunidad post-maniobra
                elif ("semaforo_verde" in clases_detectadas or "stop" in clases_detectadas or "cono" in clases_detectadas):
                    estado_actual = "AVANZA"
                    texto_pantalla = "CRUISING (INMUNIDAD)"

                elif "semaforo_rojo" in clases_detectadas:
                    estado_actual = "ALTO"
                    texto_pantalla = "SEMAFORO ROJO"

                elif "semaforo_amarillo" in clases_detectadas:
                    estado_actual = "LENTO"
                    texto_pantalla = "PRECAUCION: LENTO"

                else:
                    estado_actual = "AVANZA"
                    texto_pantalla = "CRUISING (RECTO)"

            # --- 4. Enviar instrucción al QCar ---
            conn.sendall(estado_actual.encode('utf-8'))

            # --- 5. Interfaz Visual ---
            frame_anotado = resultados[0].plot()
            cv2.putText(frame_anotado, f"ESTADO: {texto_pantalla}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Vision QCar - Autonomo", frame_anotado)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nApagando servidor...")
                break

    except Exception as e:
        print(f"\nConexión terminada. ({e})")
    finally:
        conn.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Nodo Cliente (`transmisor_qcar.py`) — Ejecutado en QCar

```python
import cv2
import socket
import struct
import numpy as np
from Quanser.pal.utilities.vision import Camera2D
from Quanser.product_QCar import QCar

MAC_IP = "192.168.0.192"  # IP del servidor (MacBook Pro)
PORT = 9999

def main():
    print(f"[+] Conectando a la Mac ({MAC_IP}:{PORT})...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client_socket.connect((MAC_IP, PORT))
        print("[+] ¡Conexión exitosa por Wi-Fi!")
    except ConnectionRefusedError:
        print("[-] ERROR: No encuentro la Mac. ¿Ya corriste el servidor_yolo.py allá?")
        return

    print("[+] Encendiendo cámara...")
    cam = Camera2D(cameraId="3", frameWidth=640, frameHeight=480, frameRate=30.0)

    print("[+] Encendiendo motores del QCar...")
    try:
        car = QCar()
    except Exception as e:
        print("Error al abrir el QCar:", e)
        return

    mtr_cmd = np.array([0.0, 0.0], dtype=np.float64)
    leds = np.zeros(8, dtype=np.float64)

    throttle = 0.0
    steer = 0.0

    print("\n--- Sistema de Navegación Autónoma Activo ---")
    print("Mapeo de cuarto 2.5m x 2.5m configurado.")
    print("Cierra el programa con Ctrl+C")

    try:
        while True:
            ok = cam.read()
            if ok:
                # Transmisión del frame
                frame = cam.imageData
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                data = buffer.tobytes()
                client_socket.sendall(struct.pack("<L", len(data)) + data)

                # Recepción del comando
                comando_recibido = client_socket.recv(1024).decode('utf-8')

                # --- TRADUCCIÓN DE ESTADOS A MOTORES ---
                if comando_recibido == "AVANZA":
                    throttle = 0.07
                    steer = 0.0
                elif comando_recibido == "LENTO":
                    throttle = 0.05
                    steer = 0.0
                elif comando_recibido == "EVADE_DER":
                    throttle = 0.07
                    steer = -0.20   # Evasión de obstáculo (suave)
                elif comando_recibido == "GIRO_IZQ":
                    throttle = 0.07
                    steer = 0.45    # Vuelta cerrada a la izquierda (Verde)
                elif comando_recibido == "GIRO_DER":
                    throttle = 0.07
                    steer = -0.45   # Vuelta cerrada a la derecha (Stop)
                elif comando_recibido == "ALTO":
                    throttle = 0.0
                    steer = 0.0

                mtr_cmd[0] = throttle
                mtr_cmd[1] = steer
                car.read_write_std(mtr_cmd, leds)

                print(f"\rEstado: {comando_recibido:<10} | Thr: {throttle:.2f} | Str: {steer:+.2f}   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nApagando sistema de tracción...")
    finally:
        # Freno automático de emergencia en cierre
        try:
            car.write_std(np.array([0.0, 0.0], dtype=np.float64), leds)
            car.terminate()
        except Exception:
            pass
        cam.terminate()
        client_socket.close()

if __name__ == "__main__":
    main()
```

## 4. Demostración Experimental en Video

A continuación se presenta la prueba final del algoritmo integrado. En el video se aprecia la telemetría en tiempo real procesada por la Mac (con YOLOv8 Nano) y las acciones físicas del QCar, demostrando el freno de emergencia ante peatones y la ejecución exitosa de las maniobras temporizadas.

<div style="text-align:center; margin-top: 2rem;">
  <iframe
    width="700"
    height="400"
    src="https://www.youtube.com/embed/dfQOy8wRuUQ"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen
    style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
  </iframe>
</div>