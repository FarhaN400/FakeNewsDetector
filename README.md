# Fake News Detector — Streamlit App

## Files
- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `fake_news_model.keras` — your trained Keras classifier (you provide this, downloaded from Colab)

## Step 1: Get the model file
In your Colab notebook, after training, run:
```python
model.save('fake_news_model.keras')
from google.colab import files
files.download('fake_news_model.keras')
```
This downloads the file to your computer. Put it in this same folder, next to `app.py`.

## Step 2: Test locally (optional but recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL it gives you (usually http://localhost:8501).

## Step 3: Push to GitHub
1. Create a new GitHub repository (public).
2. Add these 3 files to it: `app.py`, `requirements.txt`, `fake_news_model.keras`.
   - If `fake_news_model.keras` is larger than 25MB, use Git LFS:
     ```bash
     git lfs install
     git lfs track "*.keras"
     git add .gitattributes
     ```
3. Commit and push.

## Step 4: Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "New app".
3. Select your repository, branch (main), and main file path (`app.py`).
4. Click "Deploy".

First load will take a minute or two (downloading BERT weights ~440MB). After that it's cached.

## Notes
- The app re-downloads `bert-base-uncased` from Hugging Face on first run (no internet restrictions on Streamlit Cloud, so this works fine).
- Free tier has 1GB RAM — BERT-base should fit but may be slow on first request. If it crashes from memory limits, consider Hugging Face Spaces instead (more RAM available on free tier).
