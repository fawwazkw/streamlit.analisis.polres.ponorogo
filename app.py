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
import webbrowser

# Load the saved TF-IDF vectorizer for aspect prediction
aspect_vectorizer = joblib.load('tfidf_Aspek.sav')
aspect_svm = joblib.load('Aspek_Model.sav')

# Load the saved TF-IDF vectorizer for sentiment prediction
sentiment_vectorizer = joblib.load('tfidf_label_DataSekunder_2.1.sav')
sentiment_svm = joblib.load('Label_Model_DataSekunder_2.1.sav')

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
    # Calculate total text count
    total_texts = len(data)
    
    # Calculate total text count for each aspect
    total_pelayanan = len(data[data['Predicted Aspect'] == 'Pelayanan'])
    total_keamanan = len(data[data['Predicted Aspect'] == 'Keamanan'])
    total_lingkungan = len(data[data['Predicted Aspect'] == 'Lingkungan Fisik'])
    total_umum = len(data[data['Predicted Aspect'] == 'Umum'])
    
    # Calculate total positive predictions
    total_positive = len(data[data['Predicted Sentiment'] == 'Positif'])
    # Calculate total negative predictions
    total_negative = len(data[data['Predicted Sentiment'] == 'Negatif'])

    # Create a dataframe for total counts
    total_counts1 = pd.DataFrame({
        'Total Texts': [total_texts],
        'Total Positive': [total_positive],
        'Total Negative': [total_negative]
    })

    total_counts2 = pd.DataFrame({
        'Total Pelayanan': [total_pelayanan],
        'Total Keamanan': [total_keamanan],
        'Total Lingkungan Fisik': [total_lingkungan],
        'Total Umum': [total_umum]
    })

    # Display the total counts table
    st.table(total_counts1)
    st.table(total_counts2)

    # Display the predictions in a table
    st.table(data[['text', 'Predicted Aspect', 'Predicted Sentiment']])

def main():
    # Set page title and header
    st.title("Sentimen Analysis Opini Masyarakat Terhadap Polres Ponorogo")

    # Add a button to redirect to a link
    button_link = st.button("Dashboard Sentimen Analysis Opini Masyarakat Terhadap Polres Ponorogo")
    if button_link:
        # Define the link URL
        link_url = "https://public.tableau.com/app/profile/fawwaz.kumudani.widyadhana/viz/SentimenAnalysisOpiniMasyarakatTerhadapPolresPonorogo/DashboardPelayanan?publish=yes"  # Replace with your desired link

        # Open the link in a new tab
        webbrowser.open_new_tab(link_url)

    # Take user input
    option = st.selectbox("Pilih Opsi:", ("Prediksi Text", "Prediksi Banyak Teks"))

    if option == "Prediksi Text":
        user_input = st.text_input("Masukkan Text:")
        col1, col2 = st.columns([1, 7])
        prediction_button = col1.button("Prediksi")
        clear_button = col2.button("Clear")

        if prediction_button:
            # Perform prediction
            aspect_pred, sentiment_label = predict_aspect_sentiment(user_input)

            # Display predictions in a table
            predictions = pd.DataFrame({
                "Text": [user_input],
                "Prediksi Sentimen": [sentiment_label],
                "Prediksi Aspek": [aspect_pred]
            })
            st.table(predictions)

        if clear_button:
            user_input = ""

    elif option == "Prediksi Banyak Teks":
        file_path = st.file_uploader("Upload File Excel", type=["xlsx"])

        if file_path is not None:
            # Perform prediction for multiple data and get the predicted sentiments
            predicted_sentiments = predict_multiple_data(file_path)

            # Display total counts and predicted sentiments
            display_predictions(predicted_sentiments)



if __name__ == "__main__":
    main()
