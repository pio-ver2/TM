import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Estilo visual para tema de océano
st.markdown("""
    <style>
        body {
            background-color: #80d0c7;  /* Azul agua */
            color: #fff;  /* Texto blanco */
        }
        .stTitle {
            color: #0077b6;  /* Azul océano */
        }
        .stSubheader {
            color: #00b4d8;  /* Azul claro */
        }
        .stButton>button {
            background-color: #00b4d8;  /* Botón azul océano */
            color: white;  /* Texto blanco en el botón */
        }
        .stImage>div>img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Mostrar la versión de Python
st.write("🧑‍💻 **Versión de Python:**", platform.python_version())

# Cargar el modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Título principal con tema de océano
st.title("🌊 **Reconocimiento de Imágenes con YOLO** 🐚")

# Mostrar una imagen de referencia con tema oceánico
image = Image.open('OIG5.jpg')  # Cambia la imagen a una relacionada con el océano si lo deseas
st.image(image, width=350)

# Barra lateral con descripción
with st.sidebar:
    st.subheader("🏖️ **Usando un modelo entrenado en Teachable Machine**")
    st.write("""
    Puedes usar este modelo para identificar diferentes tipos de imágenes. ¡Toma una foto y ve lo que detecta el modelo!
    """)

# Captura de imagen con la cámara
img_file_buffer = st.camera_input("📸 **Toma una Foto del Océano**")

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

 
    prediction = model.predict(data)
    print(prediction)
    
    
    if prediction[0][0] > 0.5:
        st.header('🌊 **Izquierda**, con Probabilidad: ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('🏝️ **Arriba**, con Probabilidad: ' + str(prediction[0][1]))
   

# Información adicional en el pie de página
st.markdown("---")
st.caption("""
🦈 **Acerca de la aplicación**: Esta aplicación utiliza un modelo entrenado para reconocer imágenes relacionadas con el océano. 
Desarrollada con Streamlit y Keras. 🌊
""")
