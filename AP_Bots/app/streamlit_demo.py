import time
import streamlit as st
from vectordb import VectorDB
from AP_Bots.models import LLM
from datetime import datetime
from AP_Bots.app.utils import set_wide_sidebar, stream_output, get_all_bots, reset_session_state

MAX_TOKENS = 128

all_bots = get_all_bots()

if "db" not in st.session_state:
    st.session_state.db = VectorDB()

st.title("AP-Bot")

# Initialize default chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = LLM("GPT-4o-mini", gen_params={"max_new_tokens": MAX_TOKENS})

# Authentication Flow
if "logged_in" not in st.session_state:
    # Auth state management
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'
    
    st.sidebar.title("Login")
    
    # Login Panel
    if st.session_state.auth_mode == 'login':
        with st.sidebar.form("Login", clear_on_submit=True):
            username = st.text_input("Username", key="login_uname")
            password = st.text_input("Password", type="password", key="login_pwd")
            
            col1, col2 = st.columns([2, 3])
            with col1:
                login_btn = st.form_submit_button("Sign In", use_container_width=True)
            with col2:
                if st.form_submit_button("New User? Sign Up!", use_container_width=True):
                    st.session_state.auth_mode = 'signup'
                    st.rerun()

            if login_btn:
                with st.spinner("Authenticating..."):
                    success, message, user_id = st.session_state.db.authenticate_user(username, password)
                    if success:
                        st.session_state.update({
                            "logged_in": True,
                            "username": username,
                            "user_id": user_id
                        })
                        st.rerun()
                    else:
                        st.error(message)

    # Sign-up Panel
    elif st.session_state.auth_mode == 'signup':
        with st.sidebar.form("Sign Up", clear_on_submit=True):
            st.subheader("New Users")
            new_user = st.text_input("Choose Username", key="signup_uname")
            new_pass = st.text_input("Choose Password", type="password", key="signup_pwd")
            
            col1, col2 = st.columns([2, 3])
            with col1:
                signup_btn = st.form_submit_button("Create Account", use_container_width=True)
            with col2:
                if st.form_submit_button("Back to Login", use_container_width=True):
                    st.session_state.auth_mode = 'login'
                    st.rerun()

            if signup_btn:
                if not new_pass.strip():
                    st.error("Password cannot be empty")
                else:
                    with st.spinner("Creating account..."):
                        success, message, user_id = st.session_state.db.sign_up_user(new_user, new_pass)
                        if success:
                            st.session_state.update({
                                "logged_in": True,
                                "auth_mode": 'login',
                                "username": new_user,
                                "user_id": user_id
                            })
                            st.success("Account created! Logging you in...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)

# --------------------- MAIN APP INTERFACE -----------------------------
else:
    set_wide_sidebar()
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = LLM("GPT-4o-mini", gen_params={"max_new_tokens": MAX_TOKENS})

    st.sidebar.title(f"Welcome, {st.session_state.username}!")
    
    # Chat Management
    if st.sidebar.button("🧹 New Chat", use_container_width=True):
        if "conv_id" in st.session_state:
            st.session_state.db.delete_conversation(st.session_state.conv_id)
            del st.session_state["conv_id"]
        st.session_state.messages = []
        st.toast("New chat session started")

    # Bot Selection
    current_bot = st.session_state.chatbot.model_name
    if all_bots:
        with st.sidebar.expander("🤖 Chatbot Selection", expanded=True):
            try:
                default_index = all_bots.index(current_bot)
            except ValueError:
                default_index = 0
                
            selected_bot = st.selectbox(
                "Active Chatbot",
                options=all_bots,
                index=default_index,
                label_visibility="collapsed"
            )
            
            if selected_bot != current_bot:
                with st.status(f"🚀 Loading {selected_bot}...", expanded=True):
                    st.session_state.chatbot = LLM(selected_bot, gen_params={"max_new_tokens": MAX_TOKENS})
                    st.toast(f"{selected_bot} ready!", icon="🤖")
                    st.rerun()
    else:
        st.sidebar.error("No chatbots available")

    # Account Management
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Account Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Change Password", use_container_width=True):
                st.session_state["change_password"] = True
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                reset_session_state(st)
                st.rerun()
        
        if st.button("❌ Delete Account", ...):
            st.session_state.db.delete_user(st.session_state.user_id)
            reset_session_state(st, full_reset=True)
            st.success("Account deleted")
            time.sleep(1)
            st.rerun()

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Process user input
    if prompt := st.chat_input("Message AP-Bot..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.prompt_chatbot(
                    prompt=[{"role": m["role"], "content": m["content"]} 
                           for m in st.session_state.messages]
                )
                response_stream = stream_output(response)
                full_response = st.write_stream(response_stream)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Save conversation
        conversation_str = "\n".join(f"{m['role']}: {m['content']}" 
                                    for m in st.session_state.messages)
        if "conv_id" not in st.session_state:
            st.session_state.conv_id = st.session_state.db.save_conversation(
                user_id=st.session_state.user_id,
                conversation=conversation_str,
                chatbot_name=st.session_state.chatbot.model_name,
            )
        else:
            st.session_state.db.update_conversation(
                conv_id=st.session_state.conv_id,
                conversation=conversation_str,
                end_time=datetime.now().isoformat()
            )