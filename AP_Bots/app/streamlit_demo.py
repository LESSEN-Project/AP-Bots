import time

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from vectordb import initialize_db
from AP_Bots.models import LLM


def authenticate_user(username, password):
    user_collection = Collection("user")
    user_collection.load()
    res = user_collection.query(expr=f"user_name == '{username}'", output_fields=["password"])
    if not res:
        return False, "User does not exist. Please sign up."
    if res[0]['password'] != password:
        return False, "Incorrect password."
    return True, "Login successful."

def sign_up_user(username, password):
    user_collection = Collection("user")
    user_collection.load()
    res = user_collection.query(expr=f"user_name == '{username}'", output_fields=["user_name"])
    if res:
        return False, "Username already exists."
    user_id = uuid.uuid4().int >> 64
    user_collection.insert([[user_id], [username], [password]])
    return True, "Sign-up successful. Please log in."

def save_conversation(user_id, conversation, chatbot_name, embedding):
    conversation_collection = Collection("conversation")
    start_time = datetime.now().isoformat()
    conversation_collection.insert([
        [user_id],
        [conversation],
        [chatbot_name],
        [start_time],
        [datetime.now().isoformat()],
        [embedding]
    ])


def stream_output(output):

    for word in output:
        yield word
        time.sleep(0.005)

initialize_db()

st.title("AP-Bot")
chatbot = LLM("GPT-4o-mini")

# Authentication
st.sidebar.title("Login / Sign Up")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")
signup_button = st.sidebar.button("Sign Up")

if login_button:
    success, message = authenticate_user(username, password)
    if success:
        st.session_state["logged_in"] = True
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

if signup_button:
    success, message = sign_up_user(username, password)
    if success:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

if "logged_in" in st.session_state:
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
            ])
            response = stream_output(response)
            response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Save conversation
        save_conversation(username, prompt, "AP-Bot", chatbot.embed_text(prompt))
