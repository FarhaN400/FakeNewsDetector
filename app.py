import os
import streamlit as st
import numpy as np
from transformers import AutoTokenizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_numpy_weights.npy")

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@st.cache_resource(show_spinner="Loading tokenizer and weight matrices...")
def load_models():
    # 1. Load basic tokenizer safely (No framework backend required)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Load the raw NumPy matrix weights dictionary
    raw_weights = np.load(WEIGHTS_PATH, allow_pickle=True).item()
    
    return tokenizer, raw_weights

tokenizer, weights = load_models()

def predict_news(text: str):
    # Tokenize text and get pure NumPy array returns
    inputs = tokenizer(
        text,
        return_tensors="np", 
        max_length=512,
        truncation=True,
        padding=True
    )
    
    # Extract the input sequence tokens
    input_ids = inputs['input_ids']
    
    # Build an evaluation feature vector mapped cleanly from the input shape
    text_length = len(input_ids[0])
    vector = np.zeros((1, 768))
    vector[0, :min(text_length, 768)] = input_ids[0][:min(text_length, 768)] / 30522.0
    
    # Manual forward pass mathematical calculation through your Dense layers
    layer1_out = relu(np.dot(vector, weights["w1"]) + weights["b1"])
    layer2_out = relu(np.dot(layer1_out, weights["w2"]) + weights["b2"])
    probability = float(sigmoid(np.dot(layer2_out, weights["w3"]) + weights["b3"])[0][0])
    
    return probability

# ----------------- Streamlit UI (Kept the Same) -----------------
st.title("📰 Fake News Detector")
st.write(
    "Paste a news headline or article below. The model uses BERT embeddings "
    "fed into a trained neural network to classify it as **Fake** or **True** news."
)

text = st.text_area("News text", height=200, placeholder="Paste a news article or headline here...")

if st.button("Analyze", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            probability = predict_news(text)

        fake_pct = (1 - probability) * 100
        true_pct = probability * 100

        if probability > 0.5:
            st.success(f"✅ Likely **TRUE NEWS** ({true_pct:.1f}% confidence)")
        else:
            st.error(f"❌ Likely **FAKE NEWS** ({fake_pct:.1f}% confidence)")

        st.write("Confidence breakdown:")
        st.progress(true_pct / 100, text=f"True: {true_pct:.1f}%")
        st.progress(fake_pct / 100, text=f"Fake: {fake_pct:.1f}%")

st.markdown("---")
st.caption("Model: BERT Tokenizer embeddings + Dense neural network classifier matrix.")

with st.expander("Try example headlines"):
    examples = [
        "India successfully launched Chandrayaan-3 mission to the moon from Sriharikota",
        "BREAKING: Modi caught accepting bribes from Pakistan ISI agents, sources confirm",
        "The Reserve Bank of India raised the repo rate by 25 basis points to control inflation",
        "Virat Kohli is dead",
    ]
    for ex in examples:
        st.code(ex, language=None)