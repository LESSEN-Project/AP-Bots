
def ap_bot_prompt():

    return [{
        "role": "system",
        "content": "You are an adaptive, personalized assistant. You answer in a conversational style similar to the user."
    }]

def conv_title_prompt(conversation):

    return [{
        "role": "user",
        "content": f"Find a concise title for a conversation between a user and an assistant. Don't output anything except the title. Make it shorter than 7 words.\n{conversation}"
    }]