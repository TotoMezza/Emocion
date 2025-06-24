# app.py
import streamlit as st
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
from model import FERModel
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Configurar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERModel()
model.load_state_dict(torch.load("modelo_entrenado.pth", map_location=device))
model.eval().to(device)

# Emociones y colores
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (230, 57, 70),
    'Disgust': (106, 153, 78),
    'Fear': (154, 140, 152),
    'Happy': (244, 211, 94),
    'Sad': (69, 123, 157),
    'Surprise': (249, 132, 74),
    'Neutral': (168, 167, 167)
}

# Transformaciones
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Detector MTCNN
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# Streamlit
st.title("🎥 Detector de emociones en tiempo real")

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        pil_img = Image.fromarray(img[:, :, ::-1])  # BGR → RGB

        boxes, _ = mtcnn.detect(pil_img)
        draw = ImageDraw.Draw(pil_img)

        if boxes is not None:
            for box in boxes:
                face = pil_img.crop(box).convert("L").resize((48, 48))
                input_tensor = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    emotion = class_names[pred]
                    confidence = float(probs[pred]) * 100

                color = emotion_colors[emotion]
                draw.rectangle(box.tolist(), outline=color, width=3)
                draw.text((box[0], box[1] - 10), f"{emotion} ({confidence:.1f}%)", fill=color)

        return np.array(pil_img)

# Mostrar webcam en Streamlit
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
