import streamlit as st
import requests 
import config
import pickle
from transformers import AutoTokenizer

src_sent_tokenizer = AutoTokenizer.from_pretrained("google-T5/T5-base")

st.title("English to Hindi Translator")

input_sent = st.text_area("Enter your English sentence here")

if st.button("Translate"):
    if len(src_sent_tokenizer.tokenize(input_sent)) > config.Ns:
        st.warning("Please enter the english sentence having length less than 68 tokens")
    
    if input_sent.strip() == "":
        st.warning("Please enter an english sentence")
    else:
        response = requests.post("http://localhost:8000/translate",json={"text":input_sent})

        if response.status_code == 200:
            translated_text = response.json().get("hindi translation","")
            st.success("Hindi Translation: ")
            st.write(translated_text)
        else:
            st.error("API Error: Notable to perform translation",response.text)