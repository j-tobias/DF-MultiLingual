from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Optional


import torch
from transformers import pipeline




class LLAMA:
    def __init__(self):
        
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )


        self.messages = [
            {"role": "system", "content": """
        You are a helpful text modifier. Your target is to modify the provided text to invert its meaning to the opposite direction. The operation can be one of "delete", "insert" and "replace". Please generate output for the following input with 3 operations.

        [{"operation": "replace", "old_word": "great", "new_word": "terrible", "index": 4},
        {"operation": "delete", "old_word": "not", "new_word": None, "index": 17},
        {"operation": "insert", "old_word": None, "new_word": "not", "index": 24}]
        """},
        ]

    def generate(self, prompt:str):

        self.messages.append({"role": "user", "content": prompt})

        outputs = self.pipe(
            self.messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]

    def generate_changes(self, transcript:dict):

        prompt = f"""
        You are a helpful text modifier. Your target is to modify the provided text to invert its meaning to the opposite direction. The operation can be one of "delete", "insert" and "replace". Please generate output for the following input with 3 operations.

        [{"operation": "replace", "old_word": "great", "new_word": "terrible", "index": 4},
        {"operation": "delete", "old_word": "not", "new_word": None, "index": 17},
        {"operation": "insert", "old_word": None, "new_word": "not", "index": 24}]

        {transcript}
        """
        return self.generate(prompt)


SYSTEM_PROMPT = """You are an AI assistant helping to create a dataset for deepfake detection research. 
Your task is to suggest realistic text modifications that could be used in malicious content manipulation.
For each text segment, provide exactly 3 different manipulation operations that:
1. Maintain grammatical correctness
2. Change the meaning or sentiment significantly
3. Represent realistic disinformation tactics

Output format must be a JSON array with exactly 3 operations:
[{
    "operation": "replace"|"delete"|"insert",
    "old_text": string|null,
    "new_text": string|null,
    "position": int,
    "explanation": string
}]"""

# Define the output structure
class TextOperation(BaseModel):
    operation: str = Field(description="Type of operation: replace, delete, or insert")
    old_text: Optional[str] = Field(description="Text to be modified (null for insert)")
    new_text: Optional[str] = Field(description="New text (null for delete)")
    position: int = Field(description="Position in the text where modification occurs")
    explanation: str = Field(description="Explanation of why this modification is realistic")

class ModificationResponse(BaseModel):
    modifications: List[TextOperation]

# Setup LangChain components
parser = JsonOutputParser(pydantic_object=ModificationResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Modify the following text segment: {text}")
])