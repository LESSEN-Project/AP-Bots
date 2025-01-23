import time
import streamlit as st
from vectordb import VectorDB
from AP_Bots.models import LLM

if "db" not in st.session_state:
    st.session_state.db = VectorDB()

st.title("AP-Bot")

if "chatbot" not in st.session_state:
    chatbot_name = "CLAUDE-3.5-SONNET"
    st.session_state.chatbot = LLM(chatbot_name)

def stream_output(output):
    for word in output:
        yield word
        time.sleep(0.005)

# --------------------- LOGIN / SIGN-UP -------------------------
if "logged_in" not in st.session_state:
    st.sidebar.title("Login / Sign Up")
    username = st.sidebar.text_input("Username", on_change=lambda: st.session_state.update(action='login'))
    password = st.sidebar.text_input("Password", type="password", on_change=lambda: st.session_state.update(action='login'))
    login_button = st.sidebar.button("Login")
    signup_button = st.sidebar.button("Sign Up")

    if login_button or (username and password and st.session_state.get("action") == 'login'):
        success, message, user_id = st.session_state.db.authenticate_user(username, password)
        if success:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = user_id
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

    if signup_button:
        success, message, user_id = st.session_state.db.sign_up_user(username, password)
        if success:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_id"] = user_id
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

# --------------------- AFTER LOGIN -----------------------------
else:
    st.sidebar.title(f"Welcome, {st.session_state['username']}")

    # Optionally add "Clear Chat" in the sidebar with a trash can icon
    if st.sidebar.button("Clear Current Chat", icon="🗑️"):
        st.session_state.messages = []

    # Place password, logout, and delete account buttons at the bottom
    with st.sidebar.expander("Account Settings"):
        if st.button("Change Password"):
            st.session_state["change_password"] = True
            st.rerun()

        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

        if st.button("Delete Account", icon="🚨", help="This action is irreversible."):
            st.session_state.db.client.delete(
                collection_name="user",
                filter=f"user_name == '{st.session_state['username']}'"
            )
            del st.session_state["logged_in"]
            del st.session_state["username"]
            del st.session_state["user_id"]
            st.sidebar.success("Account deleted.")
            st.rerun()

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display messages in chat format
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Use default chat_input (pinned at the bottom)
    prompt = st.chat_input("What is up?")

    # If user enters a new prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = st.session_state.chatbot.prompt_chatbot(
                    prompt=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                )
                streamed = stream_output(response)
                streamed_text = st.write_stream(streamed)

        st.session_state.messages.append({"role": "assistant", "content": streamed_text})