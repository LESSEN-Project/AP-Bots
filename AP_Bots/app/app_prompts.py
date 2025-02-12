def personal_info_extraction_prompt(conversation_text, current_hobbies="", current_personality_traits=""):
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant specialized in personal information extraction. "
                "Extract only the allowed information from the conversation. "
                "Return your answer strictly as a JSON object with exactly two keys: 'user' and 'details'. "
                "The 'user' key maps to an object with the following keys: 'name', 'surname', 'age', 'profession', 'education', 'ethnicity', and 'country_of_residence'. "
                "The 'details' key maps to an object with two arrays: 'hobbies' and 'personality_traits'. "
                "For the hobbies, each item must be in continuous tense. "
                "If an extracted value consists of a noun and a verb, the verb must always come last, and all words must be combined with an underscore. "
                "For example, if the user says they like to play the guitar, the output should be 'guitar_playing'. "
                "For personality_traits, they should be single words. E.g. extraverted, shy, arrogant"
                "Additionally, if any extracted hobby or personality trait is similar to one of the current values provided, "
                "return the canonical form from the current list. "
                "Do not include any extra information. "
                "The output must be a JSON object with exactly this structure:\n"
                "{\n"
                '  "user": {\n'
                '    "name": <string>,\n'
                '    "surname": <string>,\n'
                '    "age": <number or string>,\n'
                '    "profession": <string>,\n'
                '    "education": <string>,\n'
                '    "ethnicity": <string>,\n'
                '    "country_of_residence": <string>\n'
                "  },\n"
                '  "details": {\n'
                '    "hobbies": [<string>, ...],\n'
                '    "personality_traits": [<string>, ...]\n'
                "  }\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Current canonical hobbies: {current_hobbies}\n"
                f"Current canonical personality_traits: {current_personality_traits}\n\n"
                f"Extract the allowed personal information from the following conversation:\n\n{conversation_text}"
            )
        }
    ]

def ap_bot_prompt(prev_convs, user_traits):

    return [{
        "role": "system",
        "content": 
        ("You are a conversational assistant. Answer in the language you receive the message. Use English by default.\n"
         "Snippets from user's past conversations are provided for context.\n"
        f"## Previous Conversations:\n{prev_convs}\n"
        "Also you will receive some general information about the user from a knowledge graph.\n"
        f"## User info:\n{user_traits}")
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
                    You will receive messages from past conversations, but also from the current conversation. Focus on the messages from the current conversation, and use previous conversations as a tool to understand user's overall style.
                    Return your findings as a JSON object with, at minimum, these keys:
                    {{
                    "grammar_analysis": "",
                    "vocabulary_analysis": "",
                    "average_message_length": "",
                    "tone_and_personality": "",
                    "additional_observations": "",
                    }}

                    <messages>
                    {messages}
                    </messages>

                    Be sure to maintain the specified JSON format in your final response. Each JSON key should contain a concise value."""}]
