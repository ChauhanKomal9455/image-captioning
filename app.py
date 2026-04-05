import streamlit as st
import numpy as np
import pickle
import gdown
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

st.set_page_config(page_title="AI Image Captioning")
st.title("🧠 AI Powered Image Captioning")

# 🔴 PASTE YOUR IDs HERE
MODEL_ID = "15G3juZhDZHnRJFQR2T1jUrTNG7keU4dZ"
TOKENIZER_ID = "1Q8M-NdPc2q7jIPD0kfqUMjB0zR6rKVXe"

# ---------------- LOAD EVERYTHING ----------------
@st.cache_resource
def load_all():
    model_path = "caption_model.h5"
    tokenizer_path = "tokenizer.pkl"

    # Download model
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", model_path, quiet=False)

    # Download tokenizer
    if not os.path.exists(tokenizer_path):
        gdown.download(f"https://drive.google.com/uc?id={TOKENIZER_ID}", tokenizer_path, quiet=False)

    # Load model
    model = load_model(model_path)

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load CNN (InceptionV3)
    base_model = InceptionV3(weights="imagenet")
    cnn_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.layers[-2].output
    )

    return model, tokenizer, cnn_model

model, tokenizer, cnn_model = load_all()

# 🔴 CHANGE if your value is different
max_length = 34

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(image):
    image = image.resize((299, 299))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = cnn_model.predict(image, verbose=0)
    return feature

# ---------------- WORD MAPPING ----------------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ---------------- CAPTION GENERATION ----------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = word_for_id(yhat, tokenizer)
        if word is None:
            break

        in_text += " " + word

        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Generating caption..."):
        features = extract_features(image)
        caption = generate_caption(model, tokenizer, features, max_length)

    st.success(caption)
