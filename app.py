import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Estilo visual para el tema del océano
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;  /* Azul claro del océano */
            color: #00796b;  /* Texto en color verde mar */
        }
        .stTitle {
            color: #004d40;  /* Título en verde océano oscuro */
        }
        .stSubheader {
            color: #004d40;  /* Subtítulo en verde océano oscuro */
        }
        .stButton>button {
            background-color: #00796b;  /* Botones de color verde mar */
            color: white;  /* Texto en blanco en el botón */
        }
        .stImage>div>img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSidebar {
            background-color: #b2dfdb;  /* Barra lateral con fondo de agua suave */
        }
        /* Cambiar color del texto específico a azul oscuro */
        .custom-text {
            color: #003366;  /* Azul oscuro */
        }
    </style>
""", unsafe_allow_html=True)

# Muestra la versión de Python
st.write("👨‍💻 **Versión de Python**:", platform.python_version())

# Cargar el modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título y descripción con tema oceánico
st.title("🌊 **Reconocimiento de Imágenes del Océano con YOLOv5** 🐚")
st.markdown("""
Esta aplicación utiliza un **modelo entrenado en Teachable Machine** para reconocer imágenes de objetos del océano. 
¡Captura una foto y ve lo que detecta el modelo! 📸
""")

# Barra lateral con descripción
with st.sidebar:
    st.subheader("🏖️ **Usando un modelo entrenado en Teachable Machine**")
    st.markdown("""
    <div class="custom-text">
    Puedes utilizar este modelo para detectar diferentes tipos de imágenes relacionadas con el océano y la playa. 
    Ajusta los parámetros a continuación y toma una foto para obtener la predicción.
    </div>
    """, unsafe_allow_html=True)

# Captura de imagen con la cámara
img_file_buffer = st.camera_input("📸 **Captura una Foto del Océano**")

if img_file_buffer is not None:
    # Leer la imagen capturada
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    img = Image.open(img_file_buffer)

    # Redimensionar imagen para que coincida con el tamaño que el modelo espera
    newsize = (224, 224)
    img = img.resize(newsize)
    
    # Convertir la imagen PIL a un array numpy
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Ejecutar la inferencia del modelo
    prediction = model.predict(data)
    print(prediction)
    
    # Mostrar los resultados de la predicción con una interpretación visual
    st.subheader("🔍 **Resultados de la Predicción**")

    if prediction[0][0] > 0.5:
        st.header('🌊 **Izquierda**, con Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('🏝️ **Arriba**, con Probabilidad: ' + str(prediction[0][1]))

st.markdown("---")
st.caption("""
🌊 **Acerca de la aplicación**: Esta aplicación utiliza **YOLOv5** para detección de objetos en imágenes capturadas con la cámara. 
Desarrollada con **Streamlit** y **Keras** para reconocimiento de objetos relacionados con el océano. 🐋
""")
