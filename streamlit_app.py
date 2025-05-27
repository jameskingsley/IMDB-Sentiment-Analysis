import streamlit as st
import joblib
import pandas as pd
import re
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CACHING MODEL AND VECTORIZER ==========
@st.cache_data
def load_model():
    # Load your LogisticRegression model saved with joblib
    return joblib.load('best_model_logreg.pkl')

@st.cache_data
def load_vectorizer():
    return joblib.load('vectorizer.pkl')

logreg_model = load_model()
vectorizer = load_vectorizer()

#LIME EXPLAINER 
explainer = LimeTextExplainer(class_names=['Negative', 'Neutral', 'Positive'])

#TEXT PREPROCESSING 
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

#LIME EXPLANATION
def explain_with_lime(text):
    def predict_proba_for_lime(text_list):
        # Logistic Regression accepts vectorized input and returns probabilities
        return logreg_model.predict_proba(vectorizer.transform(text_list))
    exp = explainer.explain_instance(text, predict_proba_for_lime, num_features=10)
    return exp.as_html()

#WORD CLOUD
def generate_wordcloud(texts):
    text = " ".join(texts)
    return WordCloud(width=800, height=400, background_color='white').generate(text)

#PLOTTING
def show_sentiment_counts(df):
    sentiment_counts = df['sentiment'].value_counts()

    # Bar Chart
    st.subheader('Sentiment Distribution (Bar Chart)')
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='coolwarm', ax=ax_bar)
    ax_bar.set_title('Sentiment Distribution')
    ax_bar.set_xlabel('Sentiment')
    ax_bar.set_ylabel('Count')
    st.pyplot(fig_bar)

    # Pie Chart
    st.subheader('Sentiment Distribution (Pie Chart)')
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette('coolwarm', n_colors=3), startangle=90)
    ax_pie.set_title('Sentiment Distribution')
    st.pyplot(fig_pie)

#MAIN APP
def main():
    st.title("🎬 IMDB Review Sentiment Analysis with LIME Explainability (Logistic Regression)")
    st.markdown("Upload IMDB review CSVs, visualize sentiment, generate word clouds, and explore predictions with LIME explainability.")

    uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            df = pd.read_csv(uploaded_file)

            if 'review' not in df.columns:
                st.error("The uploaded CSV must contain a 'review' column.")
                return

            df['cleaned'] = df['review'].astype(str).apply(preprocess)
            X = df['cleaned']
            predictions = logreg_model.predict(vectorizer.transform(X))
            df['sentiment'] = predictions

        st.success("Sentiment analysis complete!")

        # Show visualizations
        show_sentiment_counts(df)

        # Word Cloud Section
        st.subheader("Select a Sentiment to View Word Cloud")
        selected_sentiment = st.selectbox("Choose Sentiment:", ['Positive', 'Negative', 'Neutral'])

        filtered_reviews = df[df['sentiment'] == selected_sentiment]['review']
        if not filtered_reviews.empty:
            wordcloud = generate_wordcloud(filtered_reviews)
            st.image(wordcloud.to_array(), caption=f'{selected_sentiment} Sentiment WordCloud')
        else:
            st.warning(f"No reviews found for {selected_sentiment} sentiment.")

        # LIME Explanation
        st.subheader('LIME Explanation for a Single Review')
        review_text = st.text_area("Enter a review for explanation:")
        if review_text:
            with st.spinner("Generating explanation..."):
                explanation_html = explain_with_lime(review_text)
            pred = logreg_model.predict(vectorizer.transform([review_text]))[0]
            st.markdown(f"**Predicted Sentiment:** {pred}")
            st.components.v1.html(explanation_html, height=600, scrolling=True)

        # Download CSV Button
        st.subheader("Download Processed Data")
        output_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Output CSV",
            data=output_csv,
            file_name='sentiment_output.csv',
            mime='text/csv'
        )

if __name__ == '__main__':
    main()
