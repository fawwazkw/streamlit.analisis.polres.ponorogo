import pandas as pd
import joblib
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

# Load the saved TF-IDF vectorizer for aspect prediction
aspect_vectorizer = joblib.load('tfidf_Aspek.sav')
aspect_svm = joblib.load('Aspek_Model.sav')

# Load the saved TF-IDF vectorizer for sentiment prediction
sentiment_vectorizer = joblib.load('tfidf_label.sav')
sentiment_svm = joblib.load('Label_Model.sav')

def preprocess_text(text):
    # Remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # Remove old style text text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove commas
    text = re.sub(r',', '', text)
    # Remove numbers
    text = re.sub('[0-9]+', '', text)
    # Tokenize text
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    stemmer = PorterStemmer()
    text_tokens = tokenizer.tokenize(text)
    # Remove stopwords, and punctuation, and stem words
    nltk.download('stopwords')
    stopwords_indonesia = stopwords.words('indonesian')
    texts_clean = [stemmer.stem(word) for word in text_tokens if (word not in stopwords_indonesia and word not in string.punctuation)]
    
    return ' '.join(texts_clean)

@st.cache_data()
def predict_aspect_sentiment(user_input):
    # Preprocess the user input for aspect prediction
    aspect_input_text = pd.Series(user_input)
    aspect_input_text = aspect_input_text.apply(preprocess_text)
    aspect_input_text = aspect_vectorizer.transform(aspect_input_text)

    # Make aspect predictions
    aspect_pred = aspect_svm.predict(aspect_input_text)
    aspect_pred = aspect_pred[0]

    # Preprocess the user input for sentiment prediction
    sentiment_input_text = pd.Series(user_input)
    sentiment_input_text = sentiment_input_text.apply(preprocess_text)
    sentiment_input_text = sentiment_vectorizer.transform(sentiment_input_text)

    # Make sentiment predictions
    sentiment_pred = sentiment_svm.predict(sentiment_input_text)

    # Map sentiment predictions to labels
    sentiment_label = "Negatif" if sentiment_pred[0] == 0 else "Positif"

    return aspect_pred, sentiment_label

def predict_multiple_data(file_path):
    # Read data from Excel file
    data = pd.read_excel(file_path)

    # Create empty lists to store predictions
    aspect_predictions = []
    sentiment_labels = []

    # Iterate over each row in the data
    for index, row in data.iterrows():
        # Get the input text from the row
        user_input = row['text']

        # Perform prediction
        aspect_pred, sentiment_label = predict_aspect_sentiment(user_input)

        # Append predictions to the lists
        aspect_predictions.append(aspect_pred)
        sentiment_labels.append(sentiment_label)

    # Add the predictions to the data as new columns
    data['Predicted Aspect'] = aspect_predictions
    data['Predicted Sentiment'] = sentiment_labels

    # Return the data with the predicted columns
    return data[['text', 'Predicted Aspect', 'Predicted Sentiment']]


def display_predictions(data):
    # Display the predictions in a table
    st.table(data)

def main():
    # Set page title and header
    st.title("Text Classification Demo")
    st.header("Aspect and Sentiment Prediction")

    # Take user input
    option = st.selectbox("Select an option:", ("Single Text Prediction", "Multiple Text Prediction"))

    if option == "Single Text Prediction":
        user_input = st.text_input("Enter the text:")
        prediction_button = st.button("Predict")

        if prediction_button:
            # Perform prediction
            aspect_pred, sentiment_label = predict_aspect_sentiment(user_input)

            # Display predictions in a table
            predictions = pd.DataFrame({
                "Input Text": [user_input],
                "Predicted Sentiment": [sentiment_label],
                "Predicted Aspect": [aspect_pred]
            })
            st.table(predictions)
    elif option == "Multiple Text Prediction":
        file_path = st.file_uploader("Upload Excel file", type=["xlsx"])

        if file_path is not None:
            # Perform prediction for multiple data and get the predicted sentiments
            predicted_sentiments = predict_multiple_data(file_path)

            # Display the predicted sentiments
            st.table(predicted_sentiments)

if __name__ == "__main__":
    main()
