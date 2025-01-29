import time
import streamlit as st

from AP_Bots.app.app_prompts import conv_title_prompt

def stream_output(output):

    for word in output:
        yield word
        time.sleep(0.005)

def reset_session_state(st, full_reset=False):
    
    preserved_keys = ['db'] if not full_reset else []
    preserved = {key: st.session_state[key] for key in preserved_keys if key in st.session_state}
    st.session_state.clear()
    
    for key, value in preserved.items():
        st.session_state[key] = value
    st.session_state.auth_mode = 'login'
    
    return st

def get_conv_topic(llm, conversation):

    prompt = conv_title_prompt(conversation)
    title = llm.generate(prompt)
    title = title.strip('"')
    title = title.strip('*')

    return title