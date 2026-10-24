
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import string
from PIL import Image


# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="English Alphabet Recognition",
    page_icon="🔤",
    layout="centered"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "alphabet_cnn_best.pth"
LETTERS = string.ascii_uppercase


# -------------------------
# CNN Architecture
# Must match the training model
# -------------------------
class AlphabetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 26)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model():
    model = AlphabetCNN().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# -------------------------
# OpenCV preprocessing
# -------------------------
def preprocess_image(pil_image):
    image = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold dark strokes on a light background
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No letter detected in the image.")

    # Largest contour assumed to be the letter
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    if w * h < 10:
        raise ValueError("The image does not contain a clear letter.")

    character = binary[y:y + h, x:x + w]

    # Resize to fit within 20x20 pixels
    scale = 20 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        character,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Orientation correction found during testing
    canvas = cv2.transpose(canvas)

    # Match training normalization: [0,255] -> [-1,1]
    array = canvas.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

    return canvas, tensor.to(device)


# -------------------------
# User interface
# -------------------------
st.title("🔤 English Alphabet Recognition")

st.write(
    "Upload an image containing one handwritten English letter "
    "to predict the alphabet from A to Z."
)

st.info(
    "For best results, upload a clear, single handwritten letter "
    "on a plain, light background."
)

uploaded_file = st.file_uploader(
    "Upload a handwritten letter",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(pil_image, width=300)

        processed, tensor = preprocess_image(pil_image)

        with st.expander("View preprocessed image"):
            st.image(
                processed,
                caption="28 × 28 model input",
                width=180
            )

        model = load_model()

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

            predicted_index = probabilities.argmax().item()
            confidence = probabilities[predicted_index].item()

        predicted_letter = LETTERS[predicted_index]

        st.divider()
        st.subheader("Prediction")

        st.markdown(
            f"<div style='text-align:center;font-size:100px;"
            f"font-weight:bold'>{predicted_letter}</div>",
            unsafe_allow_html=True
        )

        st.metric(
            "Predicted Alphabet",
            predicted_letter
        )

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(float(confidence))

        if confidence < 0.60:
            st.warning(
                "Low confidence. Try uploading a clearer image."
            )

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee of correctness."
        )

    except Exception as error:
        st.error(f"Could not process image: {error}")
