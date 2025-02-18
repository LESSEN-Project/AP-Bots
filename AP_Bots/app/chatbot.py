import os
import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetComputeRunningProcesses, nvmlShutdown, nvmlDeviceGetCount

from AP_Bots.app.app_prompts import (
    conv_title_prompt,
    ap_bot_prompt,
    sent_analysis_prompt,
    style_analysis_prompt,
    personal_info_extraction_prompt
)
from AP_Bots.models import LLM
from AP_Bots.utils.output_parser import parse_json

def stream_output(output):

    for word in output:
        yield word
        time.sleep(0.005)

def get_llm(model_name="GPT-4o-mini", gen_params={"max_new_tokens": 2048, "temperature": 1}):
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
    # Assuming user_conversations is a dict with a 'metadatas' key:
    for conv in user_conversations.get("metadatas", []):
        if conv.get("title") == title:
            continue                        
        all_past_convs.append({
            "title": conv.get("title"),
            "conv": conv.get("conversation")
        })
    return all_past_convs

def sent_analysis(text):

    llm = get_llm("GPT-4o-mini", gen_params={"max_new_tokens": 128})
    prompt = sent_analysis_prompt(text)

    return parse_json(llm.generate(prompt))

def style_analysis(session_state, text):

    all_past_convs = get_conv_string(session_state.unstructured_memory)
    cur_conv = f"{all_past_convs}\nCurrent Conversation:\n\n{text}"

    llm = get_llm("GPT-4o-mini", gen_params={"max_new_tokens": 256})
    prompt = style_analysis_prompt(cur_conv)

    return llm.generate(prompt)

def extract_personal_info(session_state):

    llm = get_llm("GPT-4o", gen_params={"max_new_tokens": 512})

    conversation_text = "\n".join([msg["content"] for msg in session_state.messages if msg["role"] == "user"])
    traits = [t["p"]["name"] for t in session_state.kg.query_personality_knowledge()]
    hobbies = [t["h"]["name"] for t in session_state.kg.query_hobby_knowledge()]
    preferences = [t["p"]["name"] for t in session_state.kg.query_preference_knowledge()]

    prompt = personal_info_extraction_prompt(conversation_text, hobbies, traits, preferences)
    result = llm.generate(prompt)

    try:
        info = parse_json(result)
    except Exception as e:
        print(e)
        info = {}
    return info

def ap_bot_respond(chatbot, cur_conv, prev_convs, user_info):

    user_info_readable = format_user_knowledge(user_info)

    all_past_turns = ""
    for i, conv in enumerate(prev_convs):
        cur_turn = f"Conversation {i+1}"
        for turn in conv:
            cur_turn = f"{cur_turn}\n{turn['role']}: {turn['text']}"
        all_past_turns = f"{all_past_turns}\n{cur_turn}"

    # print(all_past_turns)
    prompt = ap_bot_prompt(all_past_turns, user_info_readable) + cur_conv 
    print(prompt)
    response = chatbot.generate(
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

def format_user_knowledge(records):

    if not records:
        return "No user knowledge records found."

    user_node = records[0]['u']
    user_name = user_node.get('name', 'User')

    def format_value(value):
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M")
        return value

    user_props_lines = []
    for key, value in user_node.items():
        user_props_lines.append(f"  {key}: {format_value(value)}")
    user_info = f"User: {user_name}\n" + "\n".join(user_props_lines)

    rel_lines = []
    for record in records:
        relationship = record['r']
        target_node = record['n']
        rel_type = relationship.type
        target_name = target_node.get('name', 'Unknown')
        rel_lines.append(f"{user_name}-[{rel_type}]->{target_name}")

    relationships_info = "Relationships:\n" + "\n".join("  " + line for line in rel_lines)

    return user_info + "\n\n" + relationships_info

