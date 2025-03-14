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

def stream_output(output):

    for word in output:
        yield word
        time.sleep(0.005)

def get_llm(model_name="GPT-4o-mini", gen_params={"max_new_tokens": 2048, "temperature": 1}):
    return LLM(model_name, gen_params=gen_params)

def sent_analysis(text):

    llm = get_llm("GPT-4o-mini", gen_params={"max_new_tokens": 128})
    prompt = sent_analysis_prompt(text)

    return llm.generate(prompt, json_output=True)

def style_analysis(session_state):

    all_past_convs = session_state.db.get_all_conversations_string(session_state.user_id, session_state.conv_id)
    cur_conv = "\n".join(f"{m['role']}: {m['content']}" for m in session_state.messages)
    cur_conv = f"{all_past_convs}\nCurrent Conversation:\n\n{cur_conv}"

    llm = get_llm("GPT-4o-mini", gen_params={"max_new_tokens": 256})
    prompt = style_analysis_prompt(cur_conv)
    result = llm.generate(prompt, json_output=True)

    return result

def extract_personal_info(session_state):

    llm = get_llm("GPT-4o", gen_params={"max_new_tokens": 512})

    conversation_text = "\n".join([msg["content"] for msg in session_state.messages if msg["role"] == "user"])
    traits = [t["p"]["name"] for t in session_state.kg.query_personality_knowledge()]
    hobbies = [t["h"]["name"] for t in session_state.kg.query_hobby_knowledge()]
    preferences = [t["p"]["name"] for t in session_state.kg.query_preference_knowledge()]

    prompt = personal_info_extraction_prompt(conversation_text, hobbies, traits, preferences)
    result = llm.generate(prompt, json_output=True)

    return result

def format_user_knowledge(records):
    if not records:
        return "No user knowledge records found."

    user_node = records[0]['u']
    user_name = user_node.get('name', 'User')

    def format_value(value):
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M")
        return value

    # Exclude specific keys
    exclude_keys = {"updated_at", "user_id", "created_at"}
    user_props_lines = []
    for key, value in user_node.items():
        if key in exclude_keys:
            continue
        user_props_lines.append(f"  {key}: {format_value(value)}")
    user_info = f"User: {user_name}\n" + "\n".join(user_props_lines)

    rel_lines = []
    for record in records:
        relationship = record['r']
        target_node = record['n']
        rel_type = relationship.type

        if rel_type == "HAS_COMMUNICATION_STYLE":
            grammar = target_node.get('grammar_analysis', '')
            vocab = target_node.get('vocabulary_analysis', '')
            tone = target_node.get('tone_and_personality', '')
            observations = target_node.get('additional_observations', '')
            style_details = []
            if grammar:
                style_details.append(f"Grammar: {grammar}")
            if vocab:
                style_details.append(f"Vocabulary: {vocab}")
            if tone:
                style_details.append(f"Tone: {tone}")
            if observations:
                style_details.append(f"Observations: {observations}")
            style_str = "; ".join(style_details)
            # Output only the content inside the parentheses
            rel_lines.append(f"{user_name}-[{rel_type}]->{style_str}")
        else:
            target_name = target_node.get('name', 'Unknown')
            rel_lines.append(f"{user_name}-[{rel_type}]->{target_name}")

    relationships_info = "Relationships:\n" + "\n".join("  " + line for line in rel_lines)

    return user_info + "\n\n" + relationships_info

def ap_bot_respond(chatbot, cur_conv, prev_convs, user_info):

    user_info_readable = format_user_knowledge(user_info)

    all_past_turns = ""
    for i, conv in enumerate(prev_convs):
        cur_turn = f"Conversation {i+1}"
        for turn in conv:
            cur_turn = f"{cur_turn}\n{turn['role']}: {turn['text']}"
        all_past_turns = f"{all_past_turns}\n{cur_turn}"

    prompt = ap_bot_prompt(all_past_turns, user_info_readable) + cur_conv 
    response = chatbot.generate(
    prompt=prompt, stream=True
    )
    if isinstance(response, str):
        response = stream_output(response)

    return response

def get_llm_config():
    return LLM.get_cfg()

def get_avail_llms():
    return [key for key in list(LLM.get_cfg().keys()) if key != "DEFAULT"]

def get_conv_topic(conversation):

    llm = get_llm(gen_params={"max_new_tokens": 32})
    prompt = conv_title_prompt(conversation)
    title = llm.generate(prompt)
    title = title.strip('"')
    title = title.strip('*')

    return title
