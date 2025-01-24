import time
import streamlit as st
from vectordb import VectorDB
from AP_Bots.models import LLM
from datetime import datetime

from AP_Bots.app.utils import set_wide_sidebar, stream_output, get_all_bots, reset_session_state, get_conv_topic

MAX_TOKENS = 128

all_bots = get_all_bots()
title_gen_bot = LLM("GPT-4o-mini", gen_params={"max_new_tokens": MAX_TOKENS//4})

if "db" not in st.session_state:
    st.session_state.db = VectorDB()

# Initialize default chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = LLM("GPT-4o-mini", gen_params={"max_new_tokens": MAX_TOKENS})

# Authentication Flow
if "logged_in" not in st.session_state:
    # Hide sidebar completely for auth pages
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Auth state management
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'

    # Centered auth container
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("AP-Bot Login")
            st.markdown("---")

            # Login Panel
            if st.session_state.auth_mode == 'login':
                with st.form("Login", clear_on_submit=True):
                    username = st.text_input("Username", key="login_uname")
                    password = st.text_input("Password", type="password", key="login_pwd")
                    
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        login_btn = st.form_submit_button("Sign In", use_container_width=True)
                    with col2:
                        if st.form_submit_button("Create Account", use_container_width=True):
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
                with st.form("Sign Up", clear_on_submit=True):
                    st.subheader("New User Registration")
                    new_user = st.text_input("Choose Username", key="signup_uname")
                    new_pass = st.text_input("Choose Password", type="password", key="signup_pwd")
                    
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        signup_btn = st.form_submit_button("Register", use_container_width=True)
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
    st.title("AP-Bot")
    
    # Chatbot Selection
    current_bot = st.session_state.chatbot.model_name
    if all_bots:
        with st.expander("🤖 Chatbot Selection", expanded=True):
            try:
                default_index = all_bots.index(current_bot)
            except ValueError:
                default_index = 0
                
            selected_bot = st.selectbox(
                "Active Chatbot",
                options=all_bots,
                index=default_index,
                key="bot_selector",
                label_visibility="collapsed"
            )
            
            if selected_bot != current_bot:
                st.session_state["pending_bot"] = selected_bot
    else:
        st.error("No chatbots available")

    # User sidebar
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.username}!")
        st.markdown("---")
        
        # Chat Management
        if st.button("🧹 New Chat", use_container_width=True):
            if "conv_id" in st.session_state:
                st.session_state.db.delete_conversation(st.session_state.conv_id)
                del st.session_state["conv_id"]
            st.session_state.messages = []
            st.toast("New chat session started")

        # Account Management
        with st.expander("⚙️ Account Settings", expanded=True):
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
            
            if st.button("❌ Delete Account", use_container_width=True):
                st.session_state.db.delete_user(st.session_state.user_id)
                reset_session_state(st, full_reset=True)
                st.success("Account deleted")
                time.sleep(1)
                st.rerun()

    # Handle model reload
    if "pending_bot" in st.session_state:
        selected_bot = st.session_state.pending_bot
        with st.status(f"🚀 Loading {selected_bot}...", expanded=True) as status:
            try:
                st.session_state.chatbot = LLM(selected_bot, gen_params={"max_new_tokens": MAX_TOKENS})
                status.update(label=f"{selected_bot} loaded successfully!", state="complete")
                del st.session_state["pending_bot"]
            except Exception as e:
                status.update(label=f"Error loading {selected_bot}", state="error")
                st.error(str(e))
                del st.session_state["pending_bot"]

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
        user_message_time = datetime.now().isoformat()
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.generate(
                    prompt=[{"role": m["role"], "content": m["content"]} 
                           for m in st.session_state.messages]
                )
                response_stream = stream_output(response)
                full_response = st.write_stream(response_stream)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Save conversation
        turn_json = {
            "start_time": user_message_time,
            "end_time": datetime.now().isoformat(),
            "chatbot_name": st.session_state.chatbot.model_name,
            "user_message": prompt,
            "assistant_message": full_response
        }

        if "conv_id" not in st.session_state:
            title = get_conv_topic(title_gen_bot, "\n".join(f"{m['role']}: {m['content']}" 
                                                        for m in st.session_state.messages))
            print(title)
            st.session_state.conv_id = st.session_state.db.save_conversation(
                user_id=st.session_state.user_id,
                turn=turn_json,
                title=title
            )
        else:
            st.session_state.db.update_conversation(
                conv_id=st.session_state.conv_id,
                turn=turn_json,
            )