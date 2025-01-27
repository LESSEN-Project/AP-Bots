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


def hide_sidebar():
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def button_style():
    st.markdown(
    """
    <style>
    /* Target all Streamlit button types including form submits */
    button[data-baseweb="button"], 
    .stButton button,
    button[type="submit"] {
        font-size: 0.8rem !important;
        padding: 0.25rem 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
    )