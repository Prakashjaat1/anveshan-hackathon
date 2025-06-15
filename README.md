# anveshan-hackathon
<br>
author : prakash jaat
# DS-1: Spot the Scam - Fraudulent Job Detector
# Author: [Your Name or Team]

import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------
# 1. Load & Clean Data
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    return text

@st.cache_data

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['description'])
    df['text'] = df['title'].fillna('') + ' ' + df['description']
    df['text'] = df['text'].apply(clean_text)
    return df

# ----------------------------
# 2. Train Model
# ----------------------------
def train_model(df):
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df['text'])
    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return clf, tfidf, f1, report

# ----------------------------
# 3. Predict on New CSV
# ----------------------------
def predict_jobs(clf, tfidf, df):
    df['text'] = df['title'].fillna('') + ' ' + df['description']
    df['text'] = df['text'].apply(clean_text)
    X_new = tfidf.transform(df['text'])
    probs = clf.predict_proba(X_new)[:,1]
    df['fraud_probability'] = probs
    df['prediction'] = (probs > 0.5).astype(int)
    return df

# ----------------------------
# 4. Streamlit Dashboard
# ----------------------------

def run_dashboard():
    st.title("🕵️ Spot the Scam - Job Fraud Detector")
    uploaded = st.file_uploader("Upload a CSV with job listings", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        sample = df.sample(min(len(df), 300))  # Limit to 300 for speed
        sample = sample.dropna(subset=['description'])

        model_data = load_and_preprocess("train_jobs.csv")
        clf, tfidf, f1, report = train_model(model_data)
        result = predict_jobs(clf, tfidf, sample)

        st.markdown(f"**Model F1 Score:** {f1:.2f}")
        st.text(report)

        st.subheader("Fraud Probability Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(result['fraud_probability'], bins=20, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Fraudulent vs Genuine Pie Chart")
        fraud_counts = result['prediction'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(fraud_counts, labels=['Genuine', 'Fraudulent'], autopct='%1.1f%%', colors=['green', 'red'])
        st.pyplot(fig2)

        st.subheader("Top 10 Most Suspicious Jobs")
        top_10 = result.sort_values(by='fraud_probability', ascending=False).head(10)
        st.dataframe(top_10[['title', 'location', 'fraud_probability']])

        st.subheader("Full Prediction Table")
        st.dataframe(result[['title', 'location', 'fraud_probability', 'prediction']])

# Run the app
if __name__ == '__main__':
    run_dashboard()
