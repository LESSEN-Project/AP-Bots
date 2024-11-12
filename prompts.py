def prepare_res_prompt(dataset, query, llm, examples, features=None, counter_examples=None, repetition_step=1):

    if dataset.name == "lamp":
        init_prompt = get_lamp_prompts(dataset.num, repetition_step)
    elif dataset.name == "amazon":
        init_prompt = amazon_prompt()
    
    if features:
        features = "\n".join(features)
    
    context = llm.prepare_context(init_prompt, f"{query}\n{features}", examples) 
    ce_examples = ""

    if counter_examples:
        i = 0
        for ce_example in counter_examples:
            ce_context = llm.prepare_context(init_prompt, f"{query}\n{features}\n{context}", ce_example) 
            if context:
                i += 1
                ce_examples = f"{ce_examples}\n<Other Writer-{i}>\n{ce_context}\n</Other Writer-{i}>\n"

    return init_prompt.format(query=query, examples=context, features=features, counter_examples=ce_examples)


def strip_all(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())    


def amazon_prompt() -> str:

    return strip_all("""You are an Amazon customer that likes to write reviews for products. You will be provided a set of features to help you understand your writing style.
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
                     </otherwriters>
                     Using the features, generate the proper review. If you haven't received some of the features, only make use of the provided ones. 
                     Only output the review and nothing else.
                     Product:
                     {query}
                     Review:""")


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

    return prompt + """\nArticle:\n{query}\nTitle:"""


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

    return prompt + """\nAbstract:\n{query}\nTitle:"""


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