---
layout: default
title: Anotación y Entrenamiento
parent: Recolección del Dataset
nav_order: 1
---

# Anotación, Entrenamiento y Exportación ONNX
{: .fs-9 }

Procesamiento del dataset en Roboflow, entrenamiento en la nube con Google Colab y optimización del modelo para hardware de borde.
{: .fs-6 .fw-300 }

---

Una vez recolectadas las imágenes con la cámara nativa del vehículo, el siguiente paso en la arquitectura fue etiquetar los datos, entrenar la red neuronal convolucional (YOLOv8) y optimizar los pesos resultantes para que la NVIDIA Jetson del QCar pudiera realizar inferencias en tiempo real sin saturar su memoria.

## 1. Transferencia y Anotación (Roboflow)

Para procesar las imágenes, se transfirió el directorio completo desde el QCar hacia la MacBook Pro mediante el protocolo SCP (Secure Copy Protocol). Posteriormente, el dataset fue cargado a la plataforma **Roboflow**.

Se realizó un proceso de anotación manual (Bounding Boxes) definiendo las 6 clases fundamentales para la rutina del vehículo autónomo:
`cono`, `persona`, `semaforo_amarillo`, `semaforo_rojo`, `semaforo_verde`, y `stop`.

## 2. Preprocesamiento y Data Augmentation

Para garantizar que el modelo generalizara correctamente y no sufriera de sobreajuste (*overfitting*), se configuró un pipeline de preparación de datos en Roboflow con los siguientes parámetros:

* **Split del Dataset:** 70% Entrenamiento / 20% Validación / 10% Prueba.
* **Preprocesamiento:** * *Auto-Orient:* Activado (para descartar metadatos de rotación incorrectos).
  * *Resize:* 640x640 (Stretch). YOLOv8 requiere entradas cuadradas; estirar la imagen asegura que la red procese el *frame* completo sin perder FOV.
* **Data Augmentation:** Se generaron versiones adicionales de cada imagen para simular condiciones reales del laboratorio:
  * *Brightness (Brillo):* Entre -25% y +25% para simular variaciones en la iluminación del techo.
  * *Blur (Desenfoque):* 1.5px para simular el desenfoque de movimiento (*motion blur*) cuando el carro gira rápidamente.
  * *Noise (Ruido):* 0.97% para emular el ruido térmico del sensor de la cámara CSI.

Finalmente, se exportó el dataset en formato **YOLOv8**.

## 3. Entrenamiento en la Nube (Google Colab)

Debido a que la Jetson Nano posee recursos limitados para entrenamiento profundo, se delegó la carga computacional a los servidores de Google Colab utilizando aceleración por GPU (Tesla T4).

Se ejecutó el siguiente script para descargar el dataset directamente desde Roboflow e iniciar el entrenamiento de 50 épocas con la arquitectura YOLOv8n (Nano), la más ligera y rápida de la familia:

```python
!pip install roboflow ultralytics

from roboflow import Roboflow
import ultralytics
ultralytics.checks()

# Descarga del dataset
rf = Roboflow(api_key="TU_API_KEY_AQUI")
project = rf.workspace("hailis-workspace-mbsnl").project("jc-qcar")
version = project.version(1)
dataset = version.download("yolov8")

# Iniciar entrenamiento de la red neuronal
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640

4. Optimización a Formato ONNX
El ecosistema nativo del QCar (Quanser APIs) funciona sobre una versión específica de Python que no es completamente compatible con las últimas dependencias de la librería ultralytics. Forzar esta instalación corría el riesgo de romper la comunicación con los motores físicos.

Para solucionar este cuello de botella arquitectónico, se tomó la decisión de exportar el modelo entrenado a ONNX (Open Neural Network Exchange). ONNX es un formato de representación matemática universal que permite cargar la red neuronal utilizando únicamente el módulo dnn de OpenCV, el cual ya viene preinstalado de forma segura en el QCar.

Python
from ultralytics import YOLO

# Cargar el modelo entrenado (pesos óptimos)
model = YOLO("runs/detect/train/weights/best.pt")

# Exportar a formato ONNX simplificado
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
5. Inferencia en Hardware de Borde (Edge Computing)
Una vez transferido el archivo best.onnx de regreso al QCar, se desarrolló el siguiente script en puro NumPy y OpenCV. Este código lee la cámara nativa, transforma la imagen a un blob tensorial, lo inyecta en la red ONNX, decodifica la matriz de predicción de salida (1, 10, 8400) y aplica Non-Maximum Suppression (NMS) para dibujar las cajas delimitadoras en tiempo real.

Python
import cv2
import numpy as np
from Quanser.pal.utilities.vision import Camera2D

def main():
    # --- CONFIGURACIÓN DEL MODELO ---
    print("[+] Cargando modelo ONNX en OpenCV...")
    # Uso del módulo DNN de cv2 preinstalado en el QCar
    net = cv2.dnn.readNet("best.onnx")

    clases = ["cono", "peaton", "semaforo_amarillo", "semaforo_rojo", "semaforo_verde", "stop"]

    # --- INICIAR CÁMARA ---
    print("[+] Iniciando cámara del QCar...")
    cam = Camera2D(cameraId="3", frameWidth=640, frameHeight=480, frameRate=30.0)

    print("\n--- Sistema Activo ---")
    print("Presiona 'q' en la ventana para salir.")

    try:
        while True:
            ok = cam.read()
            if ok:
                frame = cam.imageData.copy()
                alto_img, ancho_img, _ = frame.shape
                
                # Escalas para redimensionar las predicciones (YOLO = 640x640, Cámara = 640x480)
                x_scale = ancho_img / 640.0
                y_scale = alto_img / 640.0

                # Convertir imagen a "Blob" tensorial
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
                net.setInput(blob)
                
                # Inferencia (Forward pass)
                salida = net.forward()
                
                # Transponer matriz de predicciones (8400, 10)
                predicciones = salida[0].T 
                
                cajas = []
                confianzas = []
                ids_clases = []

                # Procesamiento de las 8400 posibles detecciones
                for fila in predicciones:
                    puntajes = fila[4:]
                    _, max_conf, _, max_loc = cv2.minMaxLoc(puntajes)
                    
                    # Umbral de confianza del 50%
                    if max_conf > 0.5:
                        id_clase = max_loc[1]
                        
                        xc, yc, w, h = fila[0], fila[1], fila[2], fila[3]
                        
                        x_min = int((xc - w/2) * x_scale)
                        y_min = int((yc - h/2) * y_scale)
                        ancho = int(w * x_scale)
                        alto = int(h * y_scale)
                        
                        cajas.append([x_min, y_min, ancho, alto])
                        confianzas.append(float(max_conf))
                        ids_clases.append(id_clase)

                # Non-Maximum Suppression (NMS) para limpiar duplicados
                indices = cv2.dnn.NMSBoxes(cajas, confianzas, 0.5, 0.4)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = cajas[i]
                        etiqueta = f"{clases[ids_clases[i]]}: {confianzas[i]*100:.0f}%"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, etiqueta, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Cerebro QCar - ONNX", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nApagando sistema...")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cam.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# Anotación, Entrenamiento y Exportación ONNX
{: .fs-9 }

Procesamiento del dataset en Roboflow, entrenamiento en la nube con Google Colab y optimización del modelo para hardware de borde.
{: .fs-6 .fw-300 }

---

Una vez recolectadas las imágenes con la cámara nativa del vehículo, el siguiente paso en la arquitectura fue etiquetar los datos, entrenar la red neuronal convolucional (YOLOv8) y optimizar los pesos resultantes para que la NVIDIA Jetson del QCar pudiera realizar inferencias en tiempo real sin saturar su memoria.

## 1. Transferencia y Anotación (Roboflow)

Para procesar las imágenes, se transfirió el directorio completo desde el QCar hacia la MacBook Pro mediante el protocolo SCP (Secure Copy Protocol). Posteriormente, el dataset fue cargado a la plataforma **Roboflow**.

Se realizó un proceso de anotación manual (Bounding Boxes) definiendo las 6 clases fundamentales para la rutina del vehículo autónomo:
`cono`, `persona`, `semaforo_amarillo`, `semaforo_rojo`, `semaforo_verde`, y `stop`.

## 2. Preprocesamiento y Data Augmentation

Para garantizar que el modelo generalizara correctamente y no sufriera de sobreajuste (*overfitting*), se configuró un pipeline de preparación de datos en Roboflow con los siguientes parámetros:

* **Split del Dataset:** 70% Entrenamiento / 20% Validación / 10% Prueba.
* **Preprocesamiento:** * *Auto-Orient:* Activado (para descartar metadatos de rotación incorrectos).
  * *Resize:* 640x640 (Stretch). YOLOv8 requiere entradas cuadradas; estirar la imagen asegura que la red procese el *frame* completo sin perder FOV.
* **Data Augmentation:** Se generaron versiones adicionales de cada imagen para simular condiciones reales del laboratorio:
  * *Brightness (Brillo):* Entre -25% y +25% para simular variaciones en la iluminación del techo.
  * *Blur (Desenfoque):* 1.5px para simular el desenfoque de movimiento (*motion blur*) cuando el carro gira rápidamente.
  * *Noise (Ruido):* 0.97% para emular el ruido térmico del sensor de la cámara CSI.

Finalmente, se exportó el dataset en formato **YOLOv8**.

## 3. Entrenamiento en la Nube (Google Colab)

Debido a que la Jetson Nano posee recursos limitados para entrenamiento profundo, se delegó la carga computacional a los servidores de Google Colab utilizando aceleración por GPU (Tesla T4).

Se ejecutó el siguiente script para descargar el dataset directamente desde Roboflow e iniciar el entrenamiento de 50 épocas con la arquitectura YOLOv8n (Nano), la más ligera y rápida de la familia:

```python
!pip install roboflow ultralytics

from roboflow import Roboflow
import ultralytics
ultralytics.checks()

# Descarga del dataset
rf = Roboflow(api_key="TU_API_KEY_AQUI")
project = rf.workspace("hailis-workspace-mbsnl").project("jc-qcar")
version = project.version(1)
dataset = version.download("yolov8")

# Iniciar entrenamiento de la red neuronal
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640
4. Optimización a Formato ONNX
El ecosistema nativo del QCar (Quanser APIs) funciona sobre una versión específica de Python que no es completamente compatible con las últimas dependencias de la librería ultralytics. Forzar esta instalación corría el riesgo de romper la comunicación con los motores físicos.

Para solucionar este cuello de botella arquitectónico, se tomó la decisión de exportar el modelo entrenado a ONNX (Open Neural Network Exchange). ONNX es un formato de representación matemática universal que permite cargar la red neuronal utilizando únicamente el módulo dnn de OpenCV, el cual ya viene preinstalado de forma segura en el QCar.

Python
from ultralytics import YOLO

# Cargar el modelo entrenado (pesos óptimos)
model = YOLO("runs/detect/train/weights/best.pt")

# Exportar a formato ONNX simplificado
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
5. Inferencia en Hardware de Borde (Edge Computing)
Una vez transferido el archivo best.onnx de regreso al QCar, se desarrolló el siguiente script en puro NumPy y OpenCV. Este código lee la cámara nativa, transforma la imagen a un blob tensorial, lo inyecta en la red ONNX, decodifica la matriz de predicción de salida (1, 10, 8400) y aplica Non-Maximum Suppression (NMS) para dibujar las cajas delimitadoras en tiempo real.

Python
import cv2
import numpy as np
from Quanser.pal.utilities.vision import Camera2D

def main():
    # --- CONFIGURACIÓN DEL MODELO ---
    print("[+] Cargando modelo ONNX en OpenCV...")
    # Uso del módulo DNN de cv2 preinstalado en el QCar
    net = cv2.dnn.readNet("best.onnx")

    clases = ["cono", "peaton", "semaforo_amarillo", "semaforo_rojo", "semaforo_verde", "stop"]

    # --- INICIAR CÁMARA ---
    print("[+] Iniciando cámara del QCar...")
    cam = Camera2D(cameraId="3", frameWidth=640, frameHeight=480, frameRate=30.0)

    print("\n--- Sistema Activo ---")
    print("Presiona 'q' en la ventana para salir.")

    try:
        while True:
            ok = cam.read()
            if ok:
                frame = cam.imageData.copy()
                alto_img, ancho_img, _ = frame.shape
                
                # Escalas para redimensionar las predicciones (YOLO = 640x640, Cámara = 640x480)
                x_scale = ancho_img / 640.0
                y_scale = alto_img / 640.0

                # Convertir imagen a "Blob" tensorial
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
                net.setInput(blob)
                
                # Inferencia (Forward pass)
                salida = net.forward()
                
                # Transponer matriz de predicciones (8400, 10)
                predicciones = salida[0].T 
                
                cajas = []
                confianzas = []
                ids_clases = []

                # Procesamiento de las 8400 posibles detecciones
                for fila in predicciones:
                    puntajes = fila[4:]
                    _, max_conf, _, max_loc = cv2.minMaxLoc(puntajes)
                    
                    # Umbral de confianza del 50%
                    if max_conf > 0.5:
                        id_clase = max_loc[1]
                        
                        xc, yc, w, h = fila[0], fila[1], fila[2], fila[3]
                        
                        x_min = int((xc - w/2) * x_scale)
                        y_min = int((yc - h/2) * y_scale)
                        ancho = int(w * x_scale)
                        alto = int(h * y_scale)
                        
                        cajas.append([x_min, y_min, ancho, alto])
                        confianzas.append(float(max_conf))
                        ids_clases.append(id_clase)

                # Non-Maximum Suppression (NMS) para limpiar duplicados
                indices = cv2.dnn.NMSBoxes(cajas, confianzas, 0.5, 0.4)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = cajas[i]
                        etiqueta = f"{clases[ids_clases[i]]}: {confianzas[i]*100:.0f}%"
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, etiqueta, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Cerebro QCar - ONNX", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nApagando sistema...")
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cam.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
