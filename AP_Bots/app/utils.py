import time
import streamlit as st
import GPUtil

from AP_Bots.models import LLM
from AP_Bots.prompts import conv_title_prompt


def stream_output(output):
    for word in output:
        yield word
        time.sleep(0.005)

def get_available_GPUmem():

    gpus = GPUtil.getGPUs()
    all_free_mem = 0
    for gpu in gpus:
        all_free_mem += gpu.memoryFree / 1024
    
    return all_free_mem

def get_all_bots():

    all_free_mem = get_available_GPUmem()
    LLM_config = LLM.get_model_cfg()
    all_models = [model for model in LLM_config.sections() if model != "DEFAULT"]
    available_models = [model for model in LLM_config.sections() if int(LLM_config[model]["min_GPU_RAM"]) < all_free_mem]

    return all_models, available_models

def reset_session_state(st, full_reset=False):
    
    preserved_keys = ['db', 'all_bots'] if not full_reset else []
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