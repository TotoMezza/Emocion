# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import pandas as pd
import numpy as np
import os
import urllib.request
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from model import FERModel  # tu modelo definido aparte

# Configuración del dispositivo y carga del modelo
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

# Detector de caras
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Título principal
st.markdown(
    """
    <h1 style='text-align:center; color:#3B4252; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;'>
        🎭 Detector de emociones faciales 😄😠😢
    </h1>
    """, unsafe_allow_html=True)

# Procesador de video
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        boxes, _ = mtcnn.detect(image_pil)

        if boxes is not None:
            draw = ImageDraw.Draw(image_pil)
            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except:
                font = ImageFont.load_default()

            for i, box in enumerate(boxes):
                face = image_pil.crop(box).convert("L").resize((48, 48))
                input_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    emotion = class_names[pred]
                    top_emotion_prob = float(probs[pred]) * 100

                color = emotion_colors.get(emotion, "red")
                draw.rectangle(box.tolist(), outline=color, width=3)
                etiqueta = f"{emotion} ({top_emotion_prob:.1f}%)"
                draw.text((box[0], box[1] - 20), etiqueta, fill=color, font=font)

            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        return image

# Stream de cámara
webrtc_streamer(
    key="realtime",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>💡 Pro tip: ¡Sonreí para que te detecte 'Happy'! 😄</p>", unsafe_allow_html=True)
