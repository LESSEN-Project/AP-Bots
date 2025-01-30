
def ap_bot_prompt(prev_convs, cur_conv):

    return [{
        "role": "system",
        "content": "You are an adaptive, personalized assistant. You answer in a conversational style similar to the user. You are allowed to use swear words as long as you don't insult the user."
    },
    {
        "role": "user",
        "content": f"""Here are the previous conversations you had with this user:
        <previous_conversations>
        {prev_convs}
        </previous_conversations>
        Analyze their conversation style, and answer their last message in the current conversation in their style. If no previous conversation is available, derive their style from the current conversation.
        <current_conversation>
        {cur_conv}
        </current_conversation>"""
    }]

def conv_title_prompt(conversation):

    return [{
        "role": "user",
        "content": f"Find a concise title for a conversation between a user and an assistant. Don't output anything except the title. Make it shorter than 7 words.\n{conversation}"
    }]

def sent_analysis_prompt(text):

    return [{
        "role": "system",
        "content": "You are a sentiment analysis model."
    },
    {
        "role": "user",
        "content": f"""Given a sentence, analyze its sentiment and return a JSON object with three scores between 0 and 1 for the categories: positive, neutral, and negative. The sum of all three scores must equal 1.
                       Input Format:
                       A single sentence provided in natural language.
                       Output Format:
                       {{
                        "positive": <float>,  
                        "neutral": <float>,  
                        "negative": <float>  
                        }}
                        Ensure that the scores are appropriately distributed based on the sentiment expressed in the input.
                        Input:
                        {text}
                        Output:"""
    }]