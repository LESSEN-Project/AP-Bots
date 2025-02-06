import configparser
import os
from threading import Thread
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from llama_cpp import Llama
from huggingface_hub import login, logging, hf_hub_download, snapshot_download
logging.set_verbosity_error()
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging, BitsAndBytesConfig, AsyncTextIteratorStreamer
logging.set_verbosity_error()

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai


class LLM:

    def __init__(self, model_name, model_params=None, gen_params=None) -> None:
        
        login(token=os.getenv("HF_API_KEY"), new_session=False)
        self.cfg = LLM.get_cfg()[model_name]
        self.model_name = model_name
        self.family = model_name.split("-")[0]
        self.repo_id = self.cfg.get("repo_id")
        self.file_name = self.cfg.get("file_name", None)
        self.context_length = int(self.cfg.get("context_length"))
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()

    @staticmethod
    def get_cfg():

        config = configparser.ConfigParser()
        config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
        return config

    def get_avail_space(self, prompt):

        avail_space = self.context_length - self.gen_params[self.name_token_var] - self.count_tokens(prompt)
        if avail_space <= 0:
            return None
        else:
            return avail_space   
        
    def trunc_chat_history(self, chat_history, hist_dedic_space=0.2):

        hist_dedic_space = int(self.context_length*0.2)
        total_hist_tokens = sum(self.count_tokens(tm['content']) for tm in chat_history)
        while total_hist_tokens > hist_dedic_space:
            removed_message = chat_history.pop(0)
            total_hist_tokens -= self.count_tokens(removed_message['content'])
        return chat_history 
       
    def count_tokens(self, prompt):

        if isinstance(prompt, list):
            prompt = "\n".join([turn["content"] for turn in prompt])
        if self.family == "GPT":
            encoding = tiktoken.encoding_for_model(self.repo_id)
            return len(encoding.encode(prompt))
        elif self.family == "GEMINI":
            return self.model.count_tokens(prompt).total_tokens
        elif self.family == "CLAUDE":
            return self.model.count_tokens(prompt)
        else:
            return len(self.tokenizer(prompt).input_ids)
        
    def prepare_context(self, prompt, context, query=None, chat_history=[]):

        if chat_history:
            chat_history = self.trunc_chat_history(chat_history)
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        query_len = self.count_tokens(query) if query else 0
        avail_space = self.get_avail_space(prompt + chat_history) - query_len  
        if avail_space:         
            while True:
                info = "\n".join([doc for doc in context])
                if self.count_tokens(info) > avail_space:
                    print("Context exceeds context window, removing one document!")
                    context = context[:-1]
                else:
                    break
            return info
        else:
            return -1
        
    def get_model_type(self):

        if self.model_name.endswith("GROQ"):
            return "GROQ"
        elif self.model_name.endswith("GGUF"):
            return "GGUF"
        elif self.model_name.endswith("DSAPI"):
            return "DSAPI"
        elif self.family in ["CLAUDE", "GPT", "GEMINI"]:
            return "proprietary"  
        else:
            return "default"
        
    def init_tokenizer(self):

        if self.model_type in ["GROQ", "GGUF", "DSAPI"]:
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif self.model_type == "proprietary":
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):

        if self.family == "GEMINI":
            self.name_token_var = "max_output_tokens"
        elif self.model_type in ["proprietary", "GGUF", "DSAPI"]:
            self.name_token_var = "max_tokens"
        else:
            self.name_token_var = "max_new_tokens"
        if gen_params is None:
            return {self.name_token_var: 512}
        if "max_new_tokens" in gen_params and self.name_token_var != "max_new_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_new_tokens")
        elif "max_tokens" in gen_params and self.name_token_var != "max_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_tokens")
        elif "max_output_tokens" in gen_params and self.name_token_var != "max_output_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_output_tokens")
        return gen_params
    
    def get_model_params(self, model_params):

        if model_params is None:
            if self.model_type == "GROQ":
                return {
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": os.getenv("GROQ_API_KEY")
                }   
            elif self.model_type == "DSAPI":
                return {
                    "base_url": "https://api.deepseek.com",
                    "api_key": os.getenv("DEEPSEEK_API_KEY")
                }                     
            elif self.family == "CLAUDE":
                return {
                    "api_key": os.getenv("ANTHROPIC_API_KEY")
                }
            elif self.family == "GPT":
                return {
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            elif self.family == "GEMINI":
                return {
                    "api_key": os.getenv("GOOGLE_API_KEY")
                }
            elif self.model_type == "GGUF":
                return {
                    "n_gpu_layers": -1,
                    "verbose": True,
                    "n_ctx": self.context_length
                }
            else:
                return {}
        else:
            return model_params
    
    def init_model(self):

        if self.family == "CLAUDE":
            return Anthropic(**self.model_params)
        elif self.family == "GPT" or self.model_type in ["GROQ", "DSAPI"]:
            return OpenAI(**self.model_params)       
        elif self.family == "GEMINI":
            genai.configure(**self.model_params)
            return genai.GenerativeModel(self.repo_id)
        elif self.model_type == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "hub")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_path = os.path.join(hf_cache_path, self.file_name)
            if not os.path.exists(model_path):
                if self.file_name.endswith("gguf"):
                    hf_hub_download(repo_id=self.repo_id, filename=self.file_name, local_dir=hf_cache_path)
                else:
                    snapshot_download(repo_id=self.repo_id, local_dir=hf_cache_path, allow_patterns = [f"*{self.file_name}*"])
            if not self.file_name.endswith("gguf"):
                len_files = len(os.listdir(model_path))
                model_path = f"{model_path}/{self.file_name}-00001-of-0000{len_files}.gguf"
            return Llama(model_path=model_path, **self.model_params)
        else: 
            bnb_config = None
            if "quantization" in self.model_params:
                quant_params = self.model_params.pop("quantization")
                if isinstance(quant_params, dict):
                    bnb_config = BitsAndBytesConfig(**quant_params)
                elif isinstance(quant_params, BitsAndBytesConfig):
                    bnb_config = quant_params
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    device_map="auto")

    def generate(self, prompt, stream=False, gen_params=None):

        if not gen_params:
            gen_params = self.gen_params
        else:
            gen_params = self.get_gen_params(gen_params)

        if self.model_type in ["GROQ", "DSAPI"] or self.family == "GPT":

            response = self.model.chat.completions.create(model=self.repo_id, messages=prompt, stream=stream, **gen_params)
            if stream:
                def stream_response():
                    has_reasoning = False
                    finished_thinking_yielded = False
                    reasoning_content = ""

                    if self.model_type == "DSAPI" and self.cfg.get("reason"):
                        yield "**Thinking**...\n\n\n"

                    for chunk in response:
                        delta = chunk.choices[0].delta
                        
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            has_reasoning = True 
                            reasoning_content += delta.reasoning_content
                            yield delta.reasoning_content
                        
                        elif hasattr(delta, 'content') and delta.content:
                            if has_reasoning and not finished_thinking_yielded:
                                yield "**\n\n\nFinished Thinking!**...\n\n\n"
                                finished_thinking_yielded = True  # Ensure it is only yielded once

                            yield delta.content
                    
                    if has_reasoning and not finished_thinking_yielded:
                        yield "**\n\n\nFinished Thinking!**...\n\n\n"

                return stream_response()
                            
            output = response.choices[0].message.content
            if self.model_type == "DSAPI" and self.cfg.get("reason"):
                reasoning_steps = response.choices[0].message.reasoning_content
                output = f"**Thinking**...\n\n\n{reasoning_steps}\n\n\n**Finished thinking!**\n\n\n{output}"

        elif self.family == "CLAUDE":

            if prompt[0]["role"] == "system":
                sys_msg = prompt[0]["content"]
                prompt = prompt[1:]
            else:
                sys_msg = ""

            if stream:
                stream = self.model.messages.stream(model=self.repo_id, messages=prompt, system=sys_msg, **gen_params).__enter__()
                return stream.text_stream
            else:
                response = self.model.messages.create(model=self.repo_id, messages=prompt, system=sys_msg, **gen_params)
                output = response.content[0].text   

        elif self.family == "GEMINI":

            messages = []
            for turn in prompt:
                role = "user" if turn["role"] in ["user", "system"] else "model"
                messages.append({
                    "role": role,
                    "parts": [turn["content"]]
                })
            response = self.model.generate_content(messages, generation_config=genai.types.GenerationConfig(**gen_params))
            output = output.text 

        else:
            
            if self.family in ["MISTRAL", "GEMMA"]:
                if len(prompt) > 1:
                    prompt = [{"role": "user", "content": "\n".join([turn["content"] for turn in prompt])}]

            if self.model_type == "GGUF":
                response = self.model.create_chat_completion(prompt, stream=False, **gen_params)
                output = response["choices"][-1]["message"]["content"]
            else:
                if stream:
                    return self.stream_hf_output(prompt, gen_params)
                else:
                    streamer = None
                pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, streamer=streamer, **gen_params)
                output = pipe(prompt)[0]["generated_text"][-1]["content"]

        return output
    
    async def stream_hf_output(self, prompt, gen_params):

        streamer = AsyncTextIteratorStreamer(self.tokenizer, skip_prompt=True)

        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, streamer=streamer, **gen_params)
        thread = Thread(target=pipe, args=(prompt,))
        thread.start()

        async for token in streamer:
            if token in ["<end_of_turn>", "<eot>", "<eos>", "<|eot_id|>", "<｜end▁of▁sentence｜>"]:
                continue
            elif token.strip() == "<think>":
                yield "**Thinking..\n\n**"
            elif token.strip() == "</think>":
                yield "**\n\nFinished Thinking!**"
            yield token