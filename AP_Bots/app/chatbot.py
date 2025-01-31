import os
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetComputeRunningProcesses, nvmlShutdown, nvmlDeviceGetCount

from AP_Bots.app.app_prompts import (
    conv_title_prompt,
    ap_bot_prompt,
    sent_analysis_prompt,
    style_analysis_prompt
)
from AP_Bots.models import LLM


def get_llm(model_name="GPT-4o-mini", gen_params={"max_new_tokens": 1024}):
    return LLM(model_name, gen_params=gen_params)

def get_conv_string(unstructured_memory, include_title=False, include_assistant=False):

    all_past_convs = ""

    for i, conv in enumerate(unstructured_memory):
        cur_turn = f"Conversation: {i}\n"
        if include_title:
            cur_turn = f"{cur_turn}Conversation Title: {conv['title']}\n"
        for turn in conv["conv"]:
            if include_assistant:
                cur_turn = f"{cur_turn}\nUser: {turn['user_message'].strip()}\nAssistant: {turn['assistant_message'].strip()}"
            else:
                cur_turn = f"{cur_turn}\n{turn['user_message'].strip()}"
    
        all_past_convs = f"{all_past_convs}\n{cur_turn}\n"

    return all_past_convs

def get_unstructured_memory(user_conversations, title):

    all_past_convs = []
    for conv in user_conversations:
        if title == conv["title"]:
            continue                        
        all_past_convs.append({
            "title": conv["title"],
            "conv": conv["conversation"]})

    return all_past_convs

def sent_analysis(text):

    llm = get_llm("GPT-4o", gen_params={"max_new_tokens": 128})
    prompt = sent_analysis_prompt(text)

    return llm.generate(prompt)

def style_analysis(session_state, text):

    all_past_convs = get_conv_string(session_state.unstructured_memory)
    print(all_past_convs)
    llm = get_llm("GPT-4o", gen_params={"max_new_tokens": 256})
    prompt = style_analysis_prompt(text)

    return llm.generate(prompt)

def ap_bot_respond(session_state):

    all_past_convs = get_conv_string(session_state.unstructured_memory, include_title=True, include_assistant=True)
    print(all_past_convs)
    cur_conv = "\n".join(f"{m['role']}: {m['content']}" for m in session_state.messages)
    prompt = ap_bot_prompt(all_past_convs, cur_conv)

    response = session_state.chatbot.generate(
    prompt=prompt, stream=True
    )
    if isinstance(response, str):
        response = stream_output(response)

    return response

def get_available_GPUmem():

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
    except Exception as e:
        print(e)
        return 0

    all_free_mem = 0
    current_process_mem = 0
    current_pid = os.getpid()

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        all_free_mem += mem_info.free / 1024**3

        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            if process.pid == current_pid:
                current_process_mem += process.usedGpuMemory / 1024**3

    nvmlShutdown()
    return round(all_free_mem + current_process_mem, 1)

def get_llm_config():
    return LLM.get_cfg()

def get_model_gpu_req():

    cfg = get_llm_config()
    return {model: int(cfg[model]["min_GPU_RAM"]) for model in cfg.sections() if model != "DEFAULT"}

def get_avail_llms():

    free_gpu_mem = get_available_GPUmem()
    model_gpu_req = get_model_gpu_req()
    available_models = [model for model in model_gpu_req.keys() if model_gpu_req[model] <= free_gpu_mem]

    return available_models, model_gpu_req, free_gpu_mem

def get_conv_topic(conversation):

    llm = get_llm(gen_params={"max_new_tokens": 32})
    prompt = conv_title_prompt(conversation)
    title = llm.generate(prompt)
    title = title.strip('"')
    title = title.strip('*')

    return title