import streamlit as st
import pickle
import pickle5 as pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

regex = re.compile('[%s]' % re.escape(string.punctuation))
wordnet = WordNetLemmatizer()
token_file = 'tokenizer.pickle'
model_list = ["Logistic Regression", "Gradient Boost Classifier"]
with open(token_file,'rb') as f:
    tokenizer = pickle.load(f)
    
@st.cache(allow_output_mutation = True)

# def Load_model(model_file):
#     model = load_model(model_file)
#     model.make_predict_function()
#     model.summary()
#     return model
    
def get_stopword_list(filename):
    with open(filename,'r',encoding = 'utf-8') as f:
        stopwords = f.readlines()
        stopset = set(m.strip() for m in stopwords)
        return list(frozenset(stopset))
stopwords = set(get_stopword_list('vietnamese.txt'))

def preprocessing_basic(text):
    text = re.sub(r'http\S+','', text)  #Loại noise(xóa link)
    text = re.sub("\\W", ' ', text)     # Xóa khoảng trắng thừa
    #Loại tokenizer và dấu câu
    token_doc = word_tokenize(text)
    result_token = []
    for i in token_doc:
        new_token = regex.sub(u'',i)
        if not new_token == u'':
            result_token.append(new_token)
    #Loại stopwords
    result_stopwords = []
    for text in result_token:
        tmp = text.split(' ')
        for i in tmp:
            if not i in stopwords :
                result_stopwords.append(i)
    #Xử lý stemming và lemmatizion
    final_doc = []
    for i in result_stopwords:
        final_doc.append(wordnet.lemmatize(i))
    return ' '.join(final_doc).lower()

if __name__ == '__main__':
    st.title('Fake news detection')
#     st.write('')
    sentence = st.text_area("Enter your news content here", height = 200)
    model_choice = st.selectbox("Select model", model_list)
    predict_btn = st.button("Predict")
    predictions = []
    
    if predict_btn:
        if sentence = '':
            st.error("")
        else:
            clean_text = []
            i = preprocessing_basic(sentence)
            clean_text.append(i)
            sequences = tokenizer.texts_to_sequences(clean_text)
            data = pad_sequences(sequences, padding= 'post', maxlen = 200)
            if model_choice == 'Logistic Regression':
                model = pickle.load(open(r"models/LR_model.pkl",'rb'))
                prediction = model.predict(clean_text)[0]
                predictions.append(prediction)

            elif model_choice == 'Gradient Boost Classifier':
                model = pickle.load(open(r"models/GBC_model.pkl",'rb'))
                prediction = model.predict(clean_text)[0]
                predictions.append(prediction)

            if predictions[0] == 0:
                st.success('This is a real news')
            if predictions[0] == 1:
                st.warning('This is a fake news')
