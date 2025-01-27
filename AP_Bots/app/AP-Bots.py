import time
from datetime import datetime
import streamlit as st
from vectordb import VectorDB

from AP_Bots.app.app_prompts import ap_bot_prompt
from AP_Bots.models import LLM
from AP_Bots.app.utils import (
    stream_output,
    get_all_bots,
    reset_session_state,
    get_conv_topic
)
from AP_Bots.app.st_css_style import set_wide_sidebar, hide_sidebar, button_style

button_style()
MAX_TITLE_TOKENS = 32
MAX_GEN_TOKENS = 256

all_bots, available_bots = get_all_bots()
title_gen_bot = LLM("GPT-4o-mini", gen_params={"max_new_tokens": MAX_TITLE_TOKENS})

# Ensure DB instance in session state
if "db" not in st.session_state:
    st.session_state.db = VectorDB()

# Initialize default chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = LLM("GPT-4o-mini", gen_params={"max_new_tokens": MAX_GEN_TOKENS})

# -------------------- AUTHENTICATION FLOW --------------------
if "logged_in" not in st.session_state:
    # Hide sidebar on login pages
    hide_sidebar()

    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'

    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.title("Login")
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
    all_bots, available_bots = get_all_bots()  # Ensure updated bot list
    if available_bots:
        with st.expander("🤖 Chatbot Selection", expanded=True):
            try:
                default_index = available_bots.index(current_bot)
            except ValueError:
                default_index = 0
                
            selected_bot = st.selectbox(
                "Active Chatbot",
                options=available_bots,
                index=default_index,
                key="bot_selector",
                label_visibility="collapsed"
            )
            
            if selected_bot != current_bot:
                st.session_state["pending_bot"] = selected_bot
    else:
        st.error("No chatbots available")

    # --------------------- SIDEBAR -----------------------------
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.username}!")

        if st.button("🚪 Logout", use_container_width=True):
            reset_session_state(st)
            st.rerun()
        
        with st.expander("⚙️ Account Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Change Password", use_container_width=True):
                    st.session_state["change_password"] = True
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("❌ Delete Account", use_container_width=True):
                    st.session_state.db.delete_user(st.session_state.user_id)
                    reset_session_state(st, full_reset=True)
                    st.success("Account deleted")
                    time.sleep(1)
                    st.rerun()

        st.markdown("---")

        # Chat Management
        if st.button("🧹 New Chat", use_container_width=True):
            if "conv_id" in st.session_state:
                del st.session_state["conv_id"]
            st.session_state.messages = []
            st.toast("New chat session started")

        # Past conversations
        user_conversations = st.session_state.db.get_all_user_convs(st.session_state.user_id)
        user_conversations = sorted(user_conversations, key = lambda x: x["end_time"], reverse=True)

        st.subheader("Past Conversations")

        for i, conv in enumerate(user_conversations):
            col1, col2 = st.columns([8, 1], gap="small")

            # Button to load conversation
            with col1:
                if st.button(conv["title"], key=f"load_conv_{conv['conv_id']}_{i}", use_container_width=True):
                    st.session_state.conv_id = conv["conv_id"]
                    loaded_messages = []
                    for turn in conv["conversation"]:
                        loaded_messages.append({"role": "user", "content": turn["user_message"]})
                        loaded_messages.append({"role": "assistant", "content": turn["assistant_message"]})
                    st.session_state.messages = loaded_messages
                    st.toast(f"Conversation '{conv['title']}' loaded.")
                    st.rerun()

            # Red trash-can button to delete conversation
            with col2:
                if st.button("🗑️", key=f"del_conv_{conv['conv_id']}_{i}",
                            help="Delete this conversation", use_container_width=True):
                    st.session_state.db.delete_conversation(conv["conv_id"])
                    # If deleting the currently loaded conversation, clear it
                    if "conv_id" in st.session_state and st.session_state.conv_id == conv["conv_id"]:
                        del st.session_state["conv_id"]
                        st.session_state.messages = []
                    st.toast(f"Conversation '{conv['title']}' deleted.")
                    st.rerun()

        st.markdown("---")

    # --------------------- MODEL RELOAD -----------------------------
    if "pending_bot" in st.session_state:
        selected_bot = st.session_state.pending_bot
        with st.status(f"🚀 Loading {selected_bot}...", expanded=True) as status:
            try:
                st.session_state.chatbot = LLM(selected_bot, gen_params={"max_new_tokens": MAX_GEN_TOKENS})
                status.update(label=f"{selected_bot} loaded successfully!", state="complete")
                del st.session_state["pending_bot"]
            except Exception as e:
                status.update(label=f"Error loading {selected_bot}", state="error")
                st.error(str(e))
                del st.session_state["pending_bot"]

    # --------------------- CHAT INTERFACE -----------------------------
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
                prompt = ap_bot_prompt() + [{"role": m["role"], "content": m["content"]} 
                           for m in st.session_state.messages]
                response = st.session_state.chatbot.generate(
                    prompt=prompt
                )
                response_stream = stream_output(response)
                full_response = st.write_stream(response_stream)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Save conversation
        turn_json = {
            "start_time": user_message_time,
            "end_time": datetime.now().isoformat(),
            "chatbot_name": st.session_state.chatbot.model_name,
            "user_message": prompt[-1]["content"],
            "assistant_message": full_response
        }

        if "conv_id" not in st.session_state:
            title = get_conv_topic(
                title_gen_bot, 
                "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
            )
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
