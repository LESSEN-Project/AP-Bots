import time
import random
import os
from datetime import datetime
from collections.abc import Iterable
import streamlit as st

from AP_Bots.utils.output_parser import parse_json
from AP_Bots.app.vectordb import VectorDB
from AP_Bots.app.chatbot import stream_output, get_avail_llms, get_conv_topic, get_llm, sent_analysis, style_analysis, ap_bot_respond, get_unstructured_memory
from AP_Bots.app.st_css_style import set_wide_sidebar, hide_sidebar, button_style, checkbox_font

button_style()
checkbox_font()

if "available_models" not in st.session_state:

    available_bots, model_gpu_req, free_gpu_mem = get_avail_llms()
    st.session_state.available_bots = available_bots
    st.session_state.model_gpu_req = model_gpu_req
    st.session_state.free_gpu_mem = free_gpu_mem
    st.session_state.current_model_gpu = 0

if "sentiment_tracker" not in st.session_state:
    st.session_state.sentiment_tracker = []

if "style_tracker" not in st.session_state:
    st.session_state.style_tracker = []

if "db" not in st.session_state:
    st.session_state.db = VectorDB()

# Initialize default chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_llm()

# -------------------- AUTHENTICATION FLOW --------------------
if "logged_in" not in st.session_state:
    # Hide sidebar on login pages
    hide_sidebar()

    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'

    with st.container():
        # Main title spanning full width
        st.title("Welcome to AP-Bots")
        
        # Create two main columns
        col_logo, col_form = st.columns([1, 1])
        
        with col_logo:
            # Logo display
            current_dir = os.path.dirname(os.path.abspath(__file__))
            logos_dir = os.path.join(current_dir, "logos")
            st.image(
                f"{logos_dir}/logo_transparent.png",
                use_container_width=True
            )

        with col_form:
            # Login/Register Form
            st.subheader("Login" if st.session_state.auth_mode == 'login' else "New User Registration")
            # st.markdown("---")
            
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
            
            elif st.session_state.auth_mode == 'signup':
                with st.form("Sign Up", clear_on_submit=True):
                    # st.subheader("New User Registration")
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

        # App explanation below both columns
        st.markdown("---")
        st.subheader("About AP-Bots")
        st.markdown("""
        - *Extended personalization*
        - *Adaptivity depending on context*
        - *Conversation history tracking and management*
        
        *Choose from various state-of-the-art AI models and start chatting!*
        """)

# --------------------- MAIN APP INTERFACE -----------------------------
else:
    set_wide_sidebar()
    st.title("AP-Bot")
    
    current_bot = st.session_state.chatbot.model_name

    with st.expander("ℹ️ Resource Information", expanded=False):
        st.info(f"""
        **Available GPU Memory:** {st.session_state.free_gpu_mem:.1f} GB
        **Current Model Usage:** {st.session_state.current_model_gpu:.1f} GB
        """)
        
        show_all_models = st.checkbox("Show all models and their GPU requirements")
        if show_all_models:
            st.write("**Model Requirements:**")
            for model in st.session_state.model_gpu_req:
                req = st.session_state.model_gpu_req.get(model, "Unknown")
                status = "🟢 Available" if model in st.session_state.available_bots else "🔴 Unavailable"
                st.write(f"- **{model}**: {req} GB ({status})")

    if st.session_state.available_bots:
        with st.expander("🤖 Chatbot Selection", expanded=True):
            # print(st.session_state.available_bots)
            default_index = st.session_state.available_bots.index(current_bot)
                
            selected_bot = st.selectbox(
                "Active Chatbot",
                options=st.session_state.available_bots,
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
            st.session_state.clear()
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
                    st.session_state.clear()
                    st.success("Account deleted")
                    time.sleep(1)
                    st.rerun()

        st.markdown("---")

        if st.button("🧹 New Chat", use_container_width=True):
            if "conv_id" in st.session_state:
                del st.session_state["conv_id"]
            st.session_state.messages = []
            st.toast("New chat session started")

        user_conversations = st.session_state.db.get_all_user_convs(st.session_state.user_id)
        st.session_state.unstructured_memory = get_unstructured_memory(user_conversations, st.session_state.get("title", None))

        if "conv_id" in st.session_state:
            current_conv_id = st.session_state.conv_id
            user_conversations = sorted(
                user_conversations,
                key=lambda x: (x["conv_id"] == current_conv_id, x["end_time"]),
                reverse=True
            )
        else:
            user_conversations = sorted(user_conversations, key=lambda x: x["end_time"], reverse=True)

        # st.subheader("Conversations")

        for i, conv in enumerate(user_conversations):
            col1, col2 = st.columns([8, 1], gap="small")

            is_current = "conv_id" in st.session_state and st.session_state.conv_id == conv["conv_id"]
            button_label = f"➡️ {conv['title']}" if is_current else conv["title"]

            # Button to load conversation
            with col1:
                if st.button(button_label, key=f"load_conv_{conv['conv_id']}_{i}", use_container_width=True):
                    st.session_state.conv_id = conv["conv_id"]
                    loaded_messages = []
                    for turn in conv["conversation"]:
                        loaded_messages.append({"role": "user", "content": turn["user_message"]})
                        loaded_messages.append({"role": "assistant", "content": turn["assistant_message"]})
                    st.session_state.messages = loaded_messages
                    st.toast(f"Conversation {conv['title']} loaded.")
                    st.rerun()
            
            # Red trash-can button to delete conversation
            with col2:
                if st.button("🗑️", key=f"del_conv_{conv['conv_id']}_{i}", help="Delete this conversation", use_container_width=True):
                    st.session_state.db.delete_conversation(conv["conv_id"])
                    
                    # **Update memory after deletion**
                    user_conversations = st.session_state.db.get_all_user_convs(st.session_state.user_id)
                    st.session_state.unstructured_memory = get_unstructured_memory(user_conversations, st.session_state.get("title", None))

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
        new_model_req = st.session_state.model_gpu_req.get(selected_bot, 0)
        
        # Calculate actual available memory including the current model's allocation
        available_mem = st.session_state.free_gpu_mem + st.session_state.current_model_gpu
        
        with st.status(f"🚀 Loading {selected_bot}...", expanded=True) as status:
            if new_model_req <= available_mem:
                
                # Load the new model
                st.session_state.chatbot = get_llm(selected_bot)
                
                # Update memory tracking
                st.session_state.current_model_gpu = new_model_req
                st.session_state.free_gpu_mem = available_mem - new_model_req
                
                # Update available bots list only once
                st.session_state.available_bots = [
                    model for model, req in st.session_state.model_gpu_req.items()
                    if req <= st.session_state.free_gpu_mem
                ]
                
                status.update(label=f"{selected_bot} loaded successfully!", state="complete")
                del st.session_state["pending_bot"]
            else:
                status.update(label=f"❌ Not enough GPU memory for {selected_bot}", state="error")
                st.error(f"Required: {new_model_req}GB, Available: {available_mem}GB")
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

        if "conv_id" not in st.session_state:
            start_time, conv_id = st.session_state.db.gen_conv_id(st.session_state.user_id)

        st.session_state.messages.append({"role": "user", "content": prompt})
        user_message_time = datetime.now().isoformat()

        with st.chat_message("user"):
            st.markdown(prompt)

        conv_id = st.session_state.conv_id if "conv_id" in st.session_state else conv_id
        search_filter = f"user_id == {st.session_state.user_id} and conv_id != {conv_id}"
        similar_turns = st.session_state.db.dense_search(prompt, search_filter)
        
        print(similar_turns)

        similar_turns = st.session_state.db.bm25_search(prompt, search_filter)
        print(similar_turns)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = ap_bot_respond(st.session_state)
                full_response = st.write_stream(response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        turn_json = {
            "start_time": user_message_time,
            "end_time": datetime.now().isoformat(),
            "chatbot_name": st.session_state.chatbot.model_name,
            "user_message": prompt,
            "assistant_message": full_response,
        }

        if "conv_id" not in st.session_state:
            st.session_state.title = get_conv_topic("\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages))
            st.session_state.conv_id = conv_id
            st.session_state.db.save_conversation(
                conv_id=conv_id,
                user_id=st.session_state.user_id,
                start_time=start_time,
                turn=turn_json,
                title=st.session_state.title
            )
            
            user_conversations = st.session_state.db.get_all_user_convs(st.session_state.user_id)
            st.session_state.unstructured_memory = get_unstructured_memory(user_conversations, st.session_state.get("title", None))
            st.rerun()
        
        else:
            st.session_state.db.update_conversation(
                conv_id=st.session_state.conv_id,
                turn=turn_json,
            )

        cur_sentiment = sent_analysis(prompt)
        st.session_state.sentiment_tracker.append(parse_json(cur_sentiment))

        cur_style = style_analysis(st.session_state, "\n".join(f"{m['content']}" for m in st.session_state.messages if m['role'] == "user"))
        st.session_state.style_tracker.append(parse_json(cur_style))