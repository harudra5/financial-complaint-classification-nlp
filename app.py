import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Page configuration
st.set_page_config(
    page_title="Financial Complaint Classifier",
    page_icon="📄",
    layout="centered"
)


# Load trained model
model = load_model(
    "attention_bigru_complaint_classifier.keras"
)


# Load tokenizer
with open("complaint_tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# Load label encoder
with open("complaint_label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# App title
st.subheader("📄 Financial Complaint Classification")

st.write(
    "Enter a financial complaint below and the AI model "
    "will predict its complaint category."
)


# User input
complaint_text = st.text_area(
    "Enter your complaint:",
    placeholder="Example: I was charged an incorrect fee on my credit card.",
    height=150
)


# Prediction button
predict_button = st.button("Predict Complaint Category")


if predict_button:

    if complaint_text.strip() == "":
        st.warning("Please enter a complaint.")

    else:
        # Convert complaint into token sequence
        sequence = tokenizer.texts_to_sequences([complaint_text])

        # Pad sequence to the same length used during training
        padded = pad_sequences(
            sequence,
            maxlen=350,
            padding="post",
            truncating="post"
        )

        # Get prediction probabilities
        probabilities = model.predict(
            padded,
            verbose=0
        )[0]

        # Get top 3 predicted classes
        top_3_indices = np.argsort(probabilities)[-3:][::-1]

        st.subheader("Top 3 Predictions")

        for rank, index in enumerate(top_3_indices, start=1):

            # Convert class number to category name
            predicted_label = label_encoder.inverse_transform(
                [index]
            )[0]

            # Get confidence
            confidence = probabilities[index] * 100

            st.write(
                f"{rank}. **{predicted_label}** — "
                f"{confidence:.2f}%"
            )