# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import pandas as pd
import os
import urllib.request
from model import FERModel  # modelo definido aparte

# Configuración del dispositivo y modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERModel()
model_path = "modelo_entrenado.pth"
url = "https://raw.githubusercontent.com/TotoMezza/Emocion/main/dev/modelo_entrenado.pth"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(url, model_path)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Emociones y colores asociados
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': '#E63946',
    'Disgust': '#6A994E',
    'Fear': '#9A8C98',
    'Happy': '#F4D35E',
    'Sad': '#457B9D',
    'Surprise': '#F9844A',
    'Neutral': '#A8A7A7'
}

# Detector de caras MTCNN
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# Transformaciones para el modelo
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Interfaz Streamlit
st.markdown(
    """
    <h1 style='text-align:center; color:#3B4252; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
        🎭 Detector de emociones faciales 😄😠😢
    </h1>
    """, unsafe_allow_html=True)

# Widgets para cargar imagen o sacar foto
uploaded_file = st.file_uploader("Subí una foto grupal (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

photo = st.camera_input("O tomá una foto con la cámara 📷")

# Prioridad: si sacaron foto, usarla. Si no, usar imagen subida.
image = None
if photo is not None:
    image = Image.open(photo).convert("RGB")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

if image is not None:
    st.image(image, caption="Imagen original", use_container_width=True)

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.warning("No se detectaron caras. 🤔 Intentá con otra foto.")
    else:
        # Ordenar cajas de izquierda a derecha para alinear con la tabla
        idx_sort = boxes[:, 0].argsort()
        boxes = boxes[idx_sort]

        st.success(f"Se detectaron {len(boxes)} cara(s) 👀")

        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=80)
        except:
            font = ImageFont.load_default()

        resultados = []

        for i, box in enumerate(boxes):
            face = image.crop(box).convert("L").resize((48, 48))
            input_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                emotion = class_names[pred]
                topk = torch.topk(probs, 3)
                top_emotions = [(class_names[idx], float(p) * 100) for idx, p in zip(topk.indices, topk.values)]

            color = emotion_colors.get(emotion, "red")

            # Dibujar caja con borde ancho
            draw.rectangle(box.tolist(), outline=color, width=5)
            etiqueta = f"Persona #{i+1}: {emotion} ({top_emotions[0][1]:.1f}%)"
            draw.text((box[0], box[1] - 30), etiqueta, fill=color, font=font)

            resultados.append((i+1, emotion, color, top_emotions))

        st.image(image, caption="Emociones detectadas 🎉", use_container_width=True)

        st.markdown("## Resultados detallados por persona")
        for persona_num, emocion_pred, color, emociones_top in resultados:
            st.markdown(f"### 🧠 Persona #{persona_num}: <span style='color:{color};'>{emocion_pred}</span>", unsafe_allow_html=True)
            df_emociones = pd.DataFrame(emociones_top, columns=["Emoción", "Confianza (%)"])
            df_emociones["Confianza (%)"] = df_emociones["Confianza (%)"].map(lambda x: f"{x:.1f}%")
            st.table(df_emociones)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>💡 Pro tip: ¡Sonreí para que te detecte 'Happy'! 😄</p>", unsafe_allow_html=True)
