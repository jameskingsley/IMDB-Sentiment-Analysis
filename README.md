# IMDB-Sentiment-Analysis


# IMDB Sentiment Analysis Project Overview

This project focuses on performing sentiment analysis on IMDB movie reviews using a **machine learning approach**. It evaluates different models such as **Logistic Regression** and **Naive Bayes**, and ultimately selects **Logistic Regression** for deployment based on its superior performance.

---

## Project Overview

The goal is to classify movie reviews into three sentiment categories:

- **Positive**
- **Neutral**
- **Negative**

We use machine learning models trained on labeled IMDB review data, with preprocessing and feature extraction done using **TF-IDF vectorization**. The app also integrates **LIME (Local Interpretable Model-agnostic Explanations)** for model explainability.

## Models Compared

- **Naive Bayes**
- **Logistic Regression**  *(Best performance)*

Logistic Regression was chosen due to:
- Higher accuracy on validation/test sets
- Better generalization to unseen reviews
- Improved performance on neutral sentiment classification

## Features

- Upload your own CSV file with IMDB reviews
- Predict sentiment using a trained Logistic Regression model
- View sentiment distribution (bar and pie charts)
- Generate word clouds for each sentiment class
- Explore model predictions using **LIME explainability**
- Download the processed results


## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib & Seaborn
- WordCloud
- LIME (for explainability)


