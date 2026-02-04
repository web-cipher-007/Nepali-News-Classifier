import streamlit as st
import joblib
import re
import pandas as pd

# 1. Load the artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('nepali_news_knn.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

knn, tfidf = load_assets()

# 2. Preprocessing
def clean_text(text):
    # Strip non-Devanagari
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return " ".join(text.split())

# 3. Streamlit UI
st.set_page_config(page_title="Nepali News Classifier", layout="centered")
st.title("🇳🇵 Nepali News Classifier")
st.markdown("---")

user_input = st.text_area("Paste Nepali news text here:", height=200)

if st.button("Predict"):
    if user_input.strip():
        # Process Input
        cleaned = clean_text(user_input)
        X_vec = tfidf.transform([cleaned])
        
        # Get Prediction and Probabilities
        prediction = knn.predict(X_vec)[0]
        probs = knn.predict_proba(X_vec)[0]
        classes = knn.classes_

        # Create a DataFrame for the chart
        prob_df = pd.DataFrame({
            'Category': classes,
            'Confidence (%)': [round(p * 100, 2) for p in probs]
        }).sort_values(by='Confidence (%)', ascending=False)

        # Result Display
        st.subheader(f"Prediction: :blue[{prediction}]")
        
        # Confidence Chart
        st.write("### Confidence Distribution")
        st.bar_chart(prob_df.set_index('Category'))
        
        # Detailed Breakdown Table
        with st.expander("See detailed scores"):
            st.table(prob_df)
            
    else:
        st.warning("Please enter some text first!")