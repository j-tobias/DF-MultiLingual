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


Your goal is to proces the transcript and return a reposnse in JSON, where you made a change to the transcript.
valid operations are: 
- "replace": replace a word or multiple coherent words in the transcript
"""

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







# llm = LLAMA()
# print("LLAMA instance created")
# response = llm.process_transcript("The old wizard gazed into his crystal ball, searching for answers.")
# print("Response: \n", response)

# try:
#     response = json.loads(response)
#     print("Parsed response:", response)
#     print("Response type:", type(response))
# except json.JSONDecodeError as e:
#     print("Failed to parse JSON:", e)



