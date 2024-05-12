import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def transform_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)

    y = []
    for i in Text:
        if i.isalnum():
            y.append(i)

    Text = y[:]
    y.clear()

    for i in Text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    Text = y[:]
    y.clear()

    for i in Text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("SMS Spam Detection")

input_sms = st.text_area("Enter the massage")

if st.button("Predict"):
    transformed_text = transform_text(input_sms)

    vector = tfidf.transform([transformed_text])

    result = model.predict(vector)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



