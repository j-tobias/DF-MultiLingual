import torch
from transformers import pipeline
from pydantic import BaseModel
from typing import List
import json


SYSTEM_PROMPT = """
System: You are a helpful assistant that always responds in JSON format following this schema:
{
    "old transcript": "string",
    "new transcript": "string",
    "change": {
        "old": "string",
        "new": "string"
    }
    "operation": "string",
}


Your goal is to proces the transcript and return a reposnse in JSON, indicating where you made a change to the transcript. Your 'change' has to be correct word by word
valid operations are: 
- "replace": replace a word or multiple coherent words in the transcript
"""

TRANSLATION_SYSTEM_PROMPT = """
You are a helpful assistant that always responds in JSON format following this schema:
<start>
{
    "original_text": "string",
    "translated_text": "string",
    "language": "string",
    "confidence": "float between 0 and 1"
}
<end>
"""

TRANSLATION_PROMPTS = {
    "en": f"Translate the following text to English and respond in JSON:\nInput: {{text}}",
    "es": f"Translate the following text to Spanish and respond in JSON:\nInput: {{text}}",
    "fr": f"Translate the following text to French and respond in JSON:\nInput: {{text}}",
    "de": f"Translate the following text to German and respond in JSON:\nInput: {{text}}",
    "it": f"Translate the following text to Italian and respond in JSON:\nInput: {{text}}",
    "pt": f"Translate the following text to Portuguese and respond in JSON:\nInput: {{text}}",
    "hi": f"Translate the following text to Hindi and respond in JSON:\nInput: {{text}}",
    "th": f"Translate the following text to Thai and respond in JSON:\nInput: {{text}}"
}

def clean_text(text: str) -> str:
    """Clean and normalize text by properly decoding unicode characters."""
    # First decode any JSON string literals
    text = text.encode('utf-8').decode('unicode_escape') if '\\u' in text else text
    # Ensure the text is properly encoded as UTF-8
    return text.encode('utf-8').decode('utf-8')

def clean_dict(d: dict) -> dict:
    """Clean and normalize dictionary by properly decoding unicode characters recursively."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned[clean_text(k)] = clean_dict(v)
        else:
            cleaned[clean_text(k)] = clean_text(str(v))
    return cleaned

def extract_text(text: str) -> str:
    """Extract content between second pair of <start> and <end> tags."""
    start_tag = "<start>"
    end_tag = "<end>"
    
    try:
        # Find first occurrence
        first_start = text.find(start_tag)
        # Find second occurrence starting after first one
        start_index = text.find(start_tag, first_start + len(start_tag))
        if start_index == -1:
            return ""
            
        start_index += len(start_tag)
        # Find end tag after second start
        end_index = text.find(end_tag, start_index)
        if end_index == -1:
            return ""
            
        return text[start_index:end_index].strip()
    except:
        return ""

class LLAMA:

    def __init__(self):
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            pad_token_id=50256,
            eos_token_id=50256,
        )
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.failed = 0

    def process_transcript(self, transcript: str) -> dict:
        """
        {
            "old transcript": "string",
            "new transcript": "string",
            "change": {
                "old": "string",
                "new": "string"
            }
            "operation": "string",
        }
        """
        self.messages.append({"role": "user", "content": transcript})
        
        outputs = self.pipe(
            self.messages,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
        )
        response = outputs[0]["generated_text"][-1]["content"].strip().split("assistant")[0]
        response = clean_text(response)

        self.messages.append({"role": "system", "content": response})
        print("LLM response:", response)
        try:
            response_dict = json.loads(response)            
            self.failed = 0
            return response_dict
        except json.JSONDecodeError as e:
            self.failed += 1
            print("Failed to parse JSON:", e)
            if self.failed <= 1:
                print(f"{self.failed}: Trying again...")
                return self.process_transcript(str(e))

    def translate(self, transcript: str, language: str) -> dict:
        """
        Translate the transcript to a different language.
        Returns a dictionary with the translation details.

        language options:
        - "en": English
        - "es": Spanish
        - "fr": French
        - "de": German
        - "it": Italian
        - "pt": Portuguese
        - "hi": Hindi
        - "th": Thai
        """
        if language not in TRANSLATION_PROMPTS:
            raise ValueError("Invalid language option.")
        
        prompt = TRANSLATION_PROMPTS[language].format(text=transcript)

        prompt = TRANSLATION_SYSTEM_PROMPT + "\n\n" + prompt

        try:
            response = self.pipe(prompt, 
                                 max_new_tokens=1000, 
                                 temperature=0.3,
                                 do_sample=True, 
                                 num_return_sequences=1)
            
            print("[translated] Response:", response[0]["generated_text"])
            response = extract_text(response[0]["generated_text"])
            print("[extracted] Response:", response)
            # response = "{" + str(response) + "}"
            translation_data = json.loads(response)
            return clean_dict(translation_data)
            
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")

# llm = LLAMA()
# print("LLAMA instance created")
# response = llm.translate("The old wizard gazed into his crystal ball, searching for answers.", "es")
# print("Translation response:", response)

