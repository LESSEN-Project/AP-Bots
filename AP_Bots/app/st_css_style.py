import streamlit as st

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