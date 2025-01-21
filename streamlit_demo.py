import streamlit as st
import time
from models import LLM

def stream_output(output):

    for word in output:
        yield word
        time.sleep(0.005)

st.title("AP-Bot")
chatbot = LLM("GPT-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        response = chatbot.prompt_chatbot(prompt=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],)
        response = stream_output(response)
        response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": response})



