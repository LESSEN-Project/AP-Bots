
def ap_bot_prompt(prev_convs, cur_conv):

    return [{
        "role": "user",
        "content": f"""Here is the current conversation you are having with the user:
        <current_conversation>
        {cur_conv}
        </current_conversation>

        For better understanding the user, here are snippets from previous conversations you had:
        <previous_conversations>
        {prev_convs}
        </previous_conversations>
        
        Give an appropriate answer to the user's message in the current conversation without explicitly mentioning the previous conversations. The previous conversation are there for context and reference only."""
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

def style_analysis_prompt(messages):
    return [{
        "role": "system",
        "content": "You are an expert in conversation style and personality analysis."
    },
    {
        "role": "user",
        "content": f"""Please analyze the messages provided below to determine the speaker’s conversation style, grammar usage, vocabulary complexity, average message length, and any other relevant features that characterize their personality and tone.

                    Return your findings as a JSON object with, at minimum, these keys:
                    {{
                    "grammar_analysis": "",
                    "vocabulary_analysis": "",
                    "average_message_length": "",
                    "tone_and_personality": "",
                    "additional_observations": ""
                    }}

                    <messages>
                    {messages}
                    </messages>

                    Be sure to maintain the specified JSON format in your final response. Each JSON key should contain a concise value."""}]
