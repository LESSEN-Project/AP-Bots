import time
import streamlit as st
from vectordb import VectorDB
from AP_Bots.models import LLM

# Initialize DB once
if "db" not in st.session_state:
    st.session_state.db = VectorDB()

st.title("AP-Bot")

# Load chatbot once
if "chatbot" not in st.session_state:
    chatbot_name = "LLAMA-3.2-3B"
    st.session_state.chatbot = LLM(chatbot_name)

def stream_output(output):
    for word in output:
        yield word
        time.sleep(0.005)

# ----------------------
# Authentication Sidebar
# ----------------------
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
else:
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    if st.sidebar.button("Change Password"):
        st.session_state["change_password"] = True
        st.rerun()
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    if st.sidebar.button("Delete Account"):
        st.session_state.db.client.delete(
            collection_name="user",
            filter=f"user_name == '{st.session_state['username']}'"
        )
        del st.session_state["logged_in"]
        del st.session_state["username"]
        del st.session_state["user_id"]
        st.sidebar.success("Account deleted.")
        st.rerun()

# ----------------------
# Main Chat Interface
# ----------------------
if "logged_in" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a container for chat messages with reduced bottom padding
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Reduced padding from 100px to 70px
        st.markdown("<div style='padding-bottom: 70px'></div>", unsafe_allow_html=True)

    # Custom CSS for fixed bottom controls
    st.write(
        """
        <style>
        /* Fixed footer container */
        .fixed-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid #e0e0e0;
            z-index: 999;
        }
        
        /* Main container layout */
        .footer-content {
            max-width: min(800px, 100% - 2rem);
            margin: 0 auto;
            display: flex;
            gap: 0.5rem;
        }
        
        /* Chat input styling */
        .footer-content .chat-input {
            flex-grow: 1;
        }
        
        /* Trash button styling */
        .footer-content .trash-button {
            margin-top: 0.5rem;
            height: 48px;
        }
        
        /* Adjust main content area */
        .main-content {
            padding-bottom: 40px !important;  /* Reduced from 120px */
        }

        @media (max-width: 768px) {
            .fixed-footer {
                padding: 0.5rem;
            }
            .footer-content {
                width: calc(100% - 1rem);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Fixed footer container
    with st.container():
        st.markdown('<div class="fixed-footer"><div class="footer-content">', unsafe_allow_html=True)
        
        # Chat input and trash button
        col1, col2 = st.columns([8, 1])
        with col1:
            prompt = st.chat_input("What is up?", key="fixed-chat-input")
        with col2:
            if st.button("🗑️", 
                       help="Clear chat history", 
                       key="trash-chat-button",
                       use_container_width=True):
                st.session_state.messages.clear()
                st.rerun()
        
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Handle user input
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
                streamed = st.write_stream(streamed)

        st.session_state.messages.append({"role": "assistant", "content": streamed})