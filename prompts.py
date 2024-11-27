def prepare_res_prompt(dataset, query, llm, examples, features=None, counter_examples=None, repetition_step=1):

    if dataset.name == "lamp":
        init_prompt = get_lamp_prompts(dataset.num, repetition_step)
        
    elif dataset.name == "amazon":
        init_prompt = amazon_prompt()
    
    feat_values = ""
    if features:
        feat_values = "\n".join(features)
    
    context = llm.prepare_context(init_prompt, examples, query=f"{query}\n{features}") 
    ce_examples = ""

    if counter_examples:
        i = 0
        for ce_example in counter_examples:
            ce_context = llm.prepare_context(init_prompt, ce_example, query=f"{query}\n{feat_values}\n{context}") 
            if context:
                i += 1
                ce_examples = f"{ce_examples}\n<Other Writer-{i}>\n{ce_context}\n</Other Writer-{i}>\n"

    return init_prompt.format(query=query, examples=context, features=feat_values, counter_examples=ce_examples)


def get_BFI_prompts(dataset, text):

    if dataset.name == "amazon":
        return amazon_BFI_analysis(text)


def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())    


def amazon_prompt(repetition_step=1) -> str:

    features = strip_all("""You are an Amazon customer that likes to write reviews for products. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar product-review pairs from your past reviews to remind you of your style:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive product-review pairs from other customers to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>""")
    
    instruction = """Using the features, generate the proper review. If you haven't received some of the features, only make use of the provided ones. Remember that ratings go from 1 to 5, 1 being the worst rating. Only output the review and nothing else."""
    instruction = "\n".join([instruction] * repetition_step)
    prompt = strip_all(f"{features}\n{instruction}\n")
    
    return prompt + """\nProduct: {query}\nReview:"""


def amazon_BFI_analysis(text):

    return [{
        "role": "system",
        "content": "You are an expert psychologist in analyzing BFI."
    },
        {
        "role": "user",
        "content": strip_all(f"""Based on the product reviews they give on Amazon, evaluate a person's personality traits according to the Big Five Inventory (BFI). Provide a score between 1 and 5 for each trait, where 1 indicates low expression and 5 indicates high expression. The traits are:
                       1. **Openness:** Reflects imagination, creativity, and a willingness to consider new ideas. High scores indicate a preference for novelty and variety, while low scores suggest a preference for routine and familiarity.
                       2. **Conscientiousness:** Pertains to organization, dependability, and discipline. High scores denote a strong sense of duty and goal-oriented behavior, whereas low scores may indicate a more spontaneous or flexible approach.
                       3. **Extraversion:** Involves sociability, assertiveness, and enthusiasm. High scores are associated with being outgoing and energetic, while low scores suggest a reserved or introverted nature.
                       4. **Agreeableness:** Relates to trustworthiness, altruism, and cooperation. High scores reflect a compassionate and friendly demeanor, whereas low scores may indicate a more competitive or challenging disposition.
                       5. **Neuroticism:** Concerns emotional stability and tendency toward negative emotions. High scores indicate a propensity for experiencing stress and mood swings, while low scores suggest calmness and emotional resilience.
                       Reviews: {text}
                       Please respond in JSON format. Each trait should be a key, and it should have two values: the score and a brief explanation.""")
    }]


def get_lamp_prompts(dataset_num: int, repetition_step) -> str:

    lamp_prompts = {
        4: _lamp_prompt_4,
        5: _lamp_prompt_5,
        7: _lamp_prompt_7
    }
    
    return lamp_prompts.get(dataset_num)(repetition_step)


def _lamp_prompt_4(repetition_step) -> str:

    features = strip_all("""You are a news editor that generates titles for articles. You will be provided a set of features to help you understand your writing style.
                    First feature you will receive is similar article-title pairs from your past works:
                    <similarpairs>
                    {examples}
                    </similarpairs>
                    Now you will receive features shedding light into how you use words and formulate sentences:
                    <features>
                    {features}
                    </features>
                    Finally, you will receive article-title pairs from other editors to help you distinguish your style from others.
                    <otherwriters>
                    {counter_examples}
                    </otherwriters>""")

    instruction = """Using the features, generate the proper title. If you haven't received some of the features, only make use of the provided ones. Only output the title and nothing else."""
    instruction = "\n".join([instruction]*repetition_step)
    prompt = strip_all(f"{features}\n{instruction}\n")

    return prompt + """\nArticle: {query}\nTitle:"""


def _lamp_prompt_5(repetition_step) -> str:

    features = strip_all("""You are a scholar that generates titles for abstracts. You will be provided a set of features to help you understand your writing style.
                     First feature you will receive is similar abstract-title pairs from your past works:
                     <similarpairs>
                     {examples}
                     </similarpairs>
                     Now you will receive features shedding light into how you use words and formulate sentences:
                     <features>
                     {features}
                     </features>
                     Finally, you will receive abstract-title pairs from other scholars to help you distinguish your style from others.
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>""")

    instruction = """Using the features, generate the proper title. If you haven't received some of the features, only make use of the provided ones. Only output the title and nothing else."""
    instruction = "\n".join([instruction]*repetition_step)
    prompt = strip_all(f"{features}\n{instruction}\n")

    return prompt + """\nAbstract: {query}\nTitle:"""


def _lamp_prompt_7(repetition_step) -> str:

    features = strip_all("""You are a Twitter user. Here is a set of your past tweets:
                     <pasttweets>
                     {examples}
                     </pasttweets>
                     Here are some features about your writing style:
                     <features>
                     {features}
                     </features>
                     Finally, here are some tweets from other users:
                     <otherwriters>
                     {counter_examples}
                     </otherwriters>
                     Now you will receive your last tweet:
                     Tweet:
                     {query}""")

    instruction = """Using the provided information, rephrase your last tweet so it better reflects your writing style. If you haven't received some of the information, only make use of the provided ones. Only output the rephrased tweet and nothing else."""
    instruction = "\n".join([instruction]*repetition_step)
    prompt = strip_all(f"{features}\n{instruction}\n")

    return prompt + """\nRephrased Tweet:"""