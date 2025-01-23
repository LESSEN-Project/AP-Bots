import time
import streamlit as st

from AP_Bots.models import LLM

def set_wide_sidebar():
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            min-width: 350px !important;
        }
        @media (max-width: 640px) {
            section[data-testid="stSidebar"] {
                min-width: 200px !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def stream_output(output):
    for word in output:
        yield word
        time.sleep(0.005)


def get_all_bots():

    return [model for model in LLM.get_model_cfg().sections() if model != "DEFAULT"]


def reset_session_state(st, full_reset=False):
    
    preserved_keys = ['db', 'all_bots'] if not full_reset else []
    
    preserved = {key: st.session_state[key] for key in preserved_keys if key in st.session_state}
    
    st.session_state.clear()
    
    for key, value in preserved.items():
        st.session_state[key] = value
    
    st.session_state.auth_mode = 'login'
    
    return st