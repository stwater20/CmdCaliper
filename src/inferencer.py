import sys

import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import torch
import openai
from sentence_transformers import SentenceTransformer


def initialize_llm_pool(llm_poo_info: list):
    llm_pool = []
    for info in llm_poo_info:
        model_name = info["model_name"]
        engine = getattr(sys.modules[__name__], info["engine_name"])(
            **info["engine_arguments"]
        )
        print(f"Initialize Engine - '{info['engine_name']}' Successfully!")
        llm_pool.append([engine, model_name])
    return llm_pool

class SentenceTransformerInferencer:
    def __init__(self, model_name, device):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.to(device)
        
        self.device = device

    def __call__(self, sentence_list: list):
        return torch.tensor(self.model.encode(sentence_list), device=self.device)

class GoogleInferenceEngine:
    def __init__(self, *args, **kwargs):
        self.client = genai.configure(api_key=kwargs["api_key"])

    def inference(self, prompt, model, temperature, max_output_tokens):
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        }

        model = genai.GenerativeModel(model)
        generation_config = genai.GenerationConfig(
            temperature=temperature, max_output_tokens=max_output_tokens
        )
        response = model.generate_content(
            prompt, generation_config=generation_config, 
            safety_settings=safety_settings
        )
        return response.text

class AnthropicInferenceEngine:
    def __init__(self, *args, **kwargs):
        self.client = anthropic.Anthropic(api_key=kwargs["api_key"])

    def inference(self, prompt, model, temperature, max_output_tokens):
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            messages=messages
        )
        return response.content[0].text

class OpenAIInferenceEngine:
    def __init__(self, *args, **kwargs):
        base_url = kwargs.get("base_url", "https://api.openai.com/v1/")
        self.client = openai.OpenAI(
            api_key=kwargs["api_key"], base_url=base_url
        )

    def inference(self, prompt, model, temperature, max_output_tokens):
        messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_output_tokens
        )
        return completion.choices[0].message.content
