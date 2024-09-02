from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-pro")
chat=model.start_chat(history=[])
def get_gemini_respone(question):
    response=chat.send_message(question,stream=True)
    return response
st.set_page_config(page_title="Question and Answer Demo")
st.header("Gemini LLM application")
if'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]
input=st.text_input("Input:",key="input")
submit=st.button("Ask Any Question")
if submit and input:
    response=get_gemini_respone(input)
    st.session_state['chat_history'].append(("you",input))
    st.header("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot",chunk.text))
st.subheader("The chat History is")
for role,text in st.session_state['chat_history']:
    st.write(f"{role}:{text}")        
