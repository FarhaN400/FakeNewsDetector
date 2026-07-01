import os
import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertModel

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model_numpy_weights.npy")

# Safe math functions for manual neural net forward pass
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@st.cache_resource(show_spinner="Loading BERT embeddings and weights...")
def load_models():
    # 1. Load Tokenizer & PyTorch BERT base
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased") # <-- Changed here too
    
    # 2. Load the raw NumPy matrix weights dictionary
    raw_weights = np.load(WEIGHTS_PATH, allow_pickle=True).item()
    
    return tokenizer, bert_model, raw_weights

tokenizer, bert_model, weights = load_models()

def predict_news(text: str):
    inputs = tokenizer(
        text,
        return_tensors="tf",
        max_length=512,
        truncation=True,
        padding=True
    )
    outputs = bert_model(**inputs)
    
    # Extract the 768-dimension vector representation from [CLS] token
    vector = outputs.last_hidden_state[:, 0, :].numpy() # Shape: (1, 768)
    
    # Manual forward pass calculation through your Neural Network layers
    # Layer 1: Dense (256) + ReLU
    layer1_out = relu(np.dot(vector, weights["w1"]) + weights["b1"])
    
    # Layer 2: Dense (128) + ReLU (Note: Dropout layers are skipped during prediction)
    layer2_out = relu(np.dot(layer1_out, weights["w2"]) + weights["b2"])
    
    # Layer 3: Dense (1) + Sigmoid 
    probability = float(sigmoid(np.dot(layer2_out, weights["w3"]) + weights["b3"])[0][0])
    
    return probability

# ----------------- Streamlit UI -----------------
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
st.caption("Model: BERT (bert-base-uncased) embeddings + Dense neural network classifier.")

with st.expander("Try example headlines"):
    examples = [
        "India successfully launched Chandrayaan-3 mission to the moon from Sriharikota",
        "BREAKING: Modi caught accepting bribes from Pakistan ISI agents, sources confirm",
        "The Reserve Bank of India raised the repo rate by 25 basis points to control inflation",
        "Virat Kohli is dead",
    ]
    for ex in examples:
        st.code(ex, language=None)