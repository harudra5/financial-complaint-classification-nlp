# financial-complaint-classification-nlp
## Project Overview

This project focuses on automatically classifying consumer financial complaints into different financial product categories using Natural Language Processing (NLP) and Deep Learning.

The project follows a progressive NLP approach, starting with traditional text representation and moving toward word embeddings, recurrent neural networks, attention mechanisms, and Transformer-based models.

The final deployed model is an **Attention-based Bidirectional GRU (BiGRU)** classifier.

##  Problem Statement

Financial institutions receive a large number of customer complaints. Manually categorizing these complaints is time-consuming and difficult to scale.

The objective of this project is to build an NLP-based classification system that can automatically identify the most relevant financial complaint category from the customer's complaint text.

## Dataset

The project uses the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Dataset**.

### Dataset Information

* Total dataset records: **17.57 million**
* Records containing complaint narratives: **3.85 million**
* Original product categories were consolidated into **13 major categories**
* Maximum **10,000 samples per category** were used for model development
* Final modeling dataset: **107,611 complaints**
* Train set: **86,088**
* Test set: **21,523**

### Target Categories

The classification system predicts one of the following 13 categories:

1. Checking or savings account
2. Credit card
3. Credit reporting or other personal consumer reports
4. Debt collection
5. Debt or credit management
6. Money transfer, virtual currency, or money service
7. Mortgage
8. Other financial service
9. Payday loan
10. Personal loan
11. Prepaid card
12. Student loan
13. Vehicle loan or lease

## NLP Pipeline

The project was developed progressively to compare different NLP approaches:

Text Data
   ↓
Preprocessing
   ↓
TF-IDF Baseline
   ↓
Word2Vec Embeddings
   ↓
RNN
   ↓
LSTM
   ↓
GRU
   ↓
BiLSTM
   ↓
BiGRU
   ↓
Attention + BiGRU
   ↓
Transformer Comparison
   ↓
Model Evaluation
   ↓
Final Model Selection
   ↓
Streamlit Deployment

## Models Compared

| Model                        | Test Accuracy |
| ---------------------------- | ------------: |
| TF-IDF + Logistic Regression |       ~17.51% |
| RNN                          |        11.86% |
| LSTM                         |        71.71% |
| GRU                          |        71.91% |
| BiLSTM                       |        74.93% |
| BiGRU                        |        76.36% |
| **Attention + BiGRU**        |    **76.53%** |
| Transformer                  |        76.70% |

The Transformer achieved a slightly higher accuracy than Attention + BiGRU, but the improvement was only **0.17 percentage points**.

Therefore, **Attention + BiGRU was selected as the final deployed model** because it provided competitive performance while maintaining a relatively lightweight architecture.

## Final Model — Attention + BiGRU

The final model architecture is:

Input Complaint
      ↓
Word2Vec Embedding
      ↓
Bidirectional GRU
      ↓
Self-Attention
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Softmax
      ↓
13 Complaint Categories

### Final Performance

* **Test Accuracy:** 76.53%
* **Macro F1-score:** 0.67
* **Weighted F1-score:** 0.76

The model performed particularly well on categories such as:

* Mortgage
* Student loan
* Money transfer
* Prepaid card
* Vehicle loan

Some minority categories were more challenging because of limited samples and semantic overlap between financial complaint categories.

## Prediction Features

The Streamlit application provides:

* Complaint text input
* Predicted complaint category
* **Top-3 predicted categories**
* Confidence score for each prediction

Example:
Top 3 Predictions

1. Credit card — 85.29%
2. Debt or credit management — 5.73%
3. Credit reporting or other personal consumer reports — 4.47%

This provides more information than simply returning a single predicted class and helps users understand the model's uncertainty.

## Streamlit Application

The trained Attention + BiGRU model is deployed using **Streamlit**.

The application allows users to enter a financial complaint and receive the predicted complaint categories along with confidence scores.

### Example Input
I was charged an incorrect fee on my credit card.

### Example Output

Credit card — 85.29%

## Project Structure

financial-complaint-classification-nlp/
│
├── app.py
├── requirements.txt
├── README.md
│
├── attention_bigru_complaint_classifier.keras
├── complaint_tokenizer.pkl
└── complaint_label_encoder.pkl

## Technologies Used

### Programming

* Python

### NLP

* NLP preprocessing
* TF-IDF
* Word2Vec
* Tokenization
* Sequence Padding

### Machine Learning

* Logistic Regression
* Scikit-learn

### Deep Learning

* TensorFlow
* Keras
* RNN
* LSTM
* GRU
* BiLSTM
* BiGRU
* Attention
* Transformer

### Deployment

* Streamlit

### Libraries
NumPy
Pandas
Scikit-learn
Gensim
TensorFlow
Matplotlib
Streamlit

## Installation

Clone the repository:
git clone https://github.com/YOUR-USERNAME/financial-complaint-classification-nlp.git

Navigate to the project directory:
cd financial-complaint-classification-nlp

Install dependencies:
pip install -r requirements.txt

Run the Streamlit application:
streamlit run app.py

## Key Learning Outcomes

Through this project, I explored the progression of NLP models from traditional text representations to modern deep learning architectures.

Key concepts implemented include:

* TF-IDF text representation
* Distributed word representations using Word2Vec
* Sequential modeling using RNN, LSTM and GRU
* Bidirectional sequence modeling
* Attention mechanisms
* Transformer-based classification
* Model comparison and evaluation
* Error analysis
* End-to-end model deployment using Streamlit

## Future Scope

Potential improvements include:

* Handling class imbalance more effectively
* Fine-tuning pretrained Transformer models such as BERT
* Multilingual and code-mixed complaint classification
* Emotion and intent detection
* Explainable AI for complaint predictions
* Integration with a production complaint-management system

## Author

**Harish Alakuntla**

Focused on **AI/ML, Data Science, NLP, Generative AI and Agentic AI**.
