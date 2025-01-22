import time

import streamlit as st
import streamlit_authenticator as stauth

from vectordb import VectorDB
from AP_Bots.models import LLM

db = VectorDB()

st.title("AP-Bot")
chatbot_name = "DeepSeek-R1-Distill-Llama-8B-GGUF"
chatbot = LLM(chatbot_name)

def stream_output(output):
    for word in output:
        yield word
        time.sleep(0.005)

if "logged_in" not in st.session_state:
    st.sidebar.title("Login / Sign Up")
    username = st.sidebar.text_input("Username", on_change=lambda: st.session_state.update(action='login'))
    password = st.sidebar.text_input("Password", type="password", on_change=lambda: st.session_state.update(action='login'))
    login_button = st.sidebar.button("Login")
    signup_button = st.sidebar.button("Sign Up")

    if login_button or (username and password and st.session_state.get("action") == 'login'):
        success, message, user_id = db.authenticate_user(username, password)
        if success:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = user_id
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

    if signup_button:
        success, message, user_id = db.sign_up_user(username, password)
        if success:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = user_id
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)
else:
    if "change_password" in st.session_state:
        st.sidebar.title("Change Password")
        current_password = st.sidebar.text_input("Current Password", type="password", key="current_pw")
        new_password = st.sidebar.text_input("New Password", type="password", key="new_pw")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_pw")
        if st.sidebar.button("Submit") or (st.session_state.get("current_pw") and st.session_state.get("new_pw") and st.session_state.get("confirm_pw") and st.session_state.get("key") == "confirm_pw"):
            if not current_password or not new_password or not confirm_password:
                st.sidebar.warning("All fields must be filled.")
            elif new_password != confirm_password:
                st.sidebar.error("Passwords do not match.")
            else:
                auth_success, _, _ = db.authenticate_user(st.session_state["username"], current_password)
                if auth_success:
                    if current_password == new_password:
                        st.sidebar.error("New password cannot be the same as the old password.")
                    else:
                        db.change_password(st.session_state["user_id"], new_password)
                        st.sidebar.success("Password changed successfully.")
                        del st.session_state["change_password"]
                        st.rerun()
                else:
                    st.sidebar.error("Current password is incorrect.")
        if st.sidebar.button("Back"):
            del st.session_state["change_password"]
            st.rerun()
    else:
        st.sidebar.title(f"Welcome, {st.session_state['username']}")
        if st.sidebar.button("Change Password"):
            st.session_state["change_password"] = True
            st.rerun()
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
        if st.sidebar.button("Delete Account"):
            db.client.delete(collection_name="user", filter=f"user_name == '{st.session_state['username']}'")
            del st.session_state["logged_in"]
            del st.session_state["username"]
            del st.session_state["user_id"]
            st.sidebar.success("Account deleted.")
            st.rerun()

# Main Chat Interface
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
