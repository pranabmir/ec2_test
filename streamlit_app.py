import streamlit as st
import requests
from scripts import s3

API_URL = "http://127.0.0.1:8000/api/v1/"
headers = {
    'Content-Type':'application/json'
}

model = st.selectbox('select model',['sentiment classifier','disaster tweet classifier'])


text = st.text_area('enter your text')
user_id = st.text_input("enter user id","example@test.com")

data = {"text":[text],"user_id":user_id}
if model=='sentiment classifier':
    model_api = 'sentiment_analysis'
elif model=='disaster tweet classifier':
    model_api = 'disaster_classifier'

if st.button('predict'):
    with st.spinner('predicting.. please wait'):
        response = requests.post(API_URL+model_api,headers = headers,
                                 json = data)
        output = response.json()
    st.write(output)