import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Preprocessing Setup ---
# Download NLTK data (only needs to be done once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# --- Load Model and Tokenizer ---
try:
    model = tf.keras.models.load_model("fake_news_lstm_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle error appropriately, maybe exit or use a dummy model
    model = None # Or raise an exception

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    print("Error: tokenizer.pkl not found. Please ensure the file exists.")
    tokenizer = None # Or raise an exception
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None # Or raise an exception

# Constants
max_len = 200

# --- Text Preprocessing Function (Mirrors Notebook) ---
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords and stem
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# --- Prediction Function (With Preprocessing) ---
def predict_news(text):
    if not model or not tokenizer:
        return "Error: Model or Tokenizer not loaded properly."
    if not text or not isinstance(text, str) or text.strip() == "":
        return "Please enter some news text."

    try:
        # Apply the same preprocessing as in the notebook
        processed_text = preprocess_text(text)

        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([processed_text])
        pad = pad_sequences(seq, maxlen=max_len)

        # Predict
        pred = model.predict(pad)[0][0]

        # Return result
        if pred > 0.5: # Using 0.5 as threshold, adjust if needed based on notebook validation
            return "✅ Real News"
        else:
            return "🛑 Fake News"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error during prediction: {e}"

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 📰 Fake News Detection App")
    gr.Markdown("Enter a news article below to classify it as **Fake** or **Real**.")

    input_text = gr.Textbox(lines=10, placeholder="Enter the news text here...") # Increased lines for better visibility
    output_label = gr.Label()
    submit_btn = gr.Button("Classify")
    clear_btn = gr.Button("Reset")

    submit_btn.click(fn=predict_news, inputs=input_text, outputs=output_label)
    # Clear function needs to return empty values for all outputs it clears
    clear_btn.click(fn=lambda: ("", None), inputs=None, outputs=[input_text, output_label], queue=False)

# --- Launch the App ---
if __name__ == "__main__":
    if model and tokenizer:
        print("Launching Gradio App...")
        demo.launch() # Share=True can be added if public access is needed temporarily
    else:
        print("Could not launch Gradio app due to loading errors.")

