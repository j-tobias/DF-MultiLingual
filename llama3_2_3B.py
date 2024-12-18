from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional


import torch
from transformers import pipeline


SYSTEM_PROMPT = """
You are a helpful text modifier. Your target is to modify the provided text. Generate exactly 3 operations.
The output should be a valid JSON object with a 'modifications' array containing the operations.

Each operation must have:
- "operation": one of "delete", "insert", or "replace"
- "old_text": text to modify (null for insert)
- "new_text": new text (null for delete)
- "position": integer position in text
- "explanation": string explaining the modification

Example output:
{"modifications": [
    {"operation": "replace", "old_text": "great", "new_text": "terrible", "position": 4, "explanation": "Changed sentiment"},
    {"operation": "delete", "old_text": "not", "new_text": null, "position": 17, "explanation": "Removed negation"},
    {"operation": "insert", "old_text": null, "new_text": "not", "position": 24, "explanation": "Added negation"}
]}
"""

class LLAMA:
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=2,  # Set pad_token_id explicitly
            eos_token_id=2,  # Set eos_token_id explicitly
        )

        self.parser = PydanticOutputParser(pydantic_object=ModificationResponse)
        self.format_instructions = self.parser.get_format_instructions()

    def process_transcript(self, transcript: str):
        # Create a more structured prompt
        prompt = f"""{SYSTEM_PROMPT}
{self.format_instructions}

Text to modify: {transcript}

Remember to respond with a valid JSON object only.
"""
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            return_full_text=False,
            do_sample=True,
            num_return_sequences=1,
        )
        
        try:
            response_text = outputs[0]["generated_text"]
            # Find the first occurrence of a valid JSON object
            start_idx = response_text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON object found in response")
            
            json_str = response_text[start_idx:]
            # Balance the brackets
            bracket_count = 0
            end_idx = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                raise ValueError("Unbalanced JSON object")
                
            json_str = json_str[:end_idx]
            
            # Create a default response if parsing fails
            if "modifications" not in json_str:
                json_str = '{"modifications": []}'
            
            parsed_output = self.parser.parse(json_str)
            return parsed_output
            
        except Exception as e:
            print(f"Error parsing output: {e}")
            # Return a valid empty response instead of None
            return ModificationResponse(modifications=[])


# Define the output structure
class TextOperation(BaseModel):
    model_config = {
        "json_schema_extra": {
            "properties": {
                "operation": {"description": "Type of operation: replace, delete, or insert"},
                "old_text": {"description": "Text to be modified (null for insert)"},
                "new_text": {"description": "New text (null for delete)"},
                "position": {"description": "Position in the text where modification occurs"},
                "explanation": {"description": "Explanation of why this modification is realistic"}
            }
        }
    }
    
    operation: str
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    position: int
    explanation: str

class ModificationResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "properties": {
                "modifications": {"description": "List of text modifications"}
            }
        }
    }
    
    modifications: List[TextOperation]



import json

class TextModifier:
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    def get_modifications(self, text: str) -> list:
        """
        Generate modifications for the input text.
        Returns a list of modification dictionaries.
        """
        prompt = f"""
        Modify the following text. Generate exactly 3 modifications.
        Each modification should be one of: replace, delete, or insert.
        
        Text to modify: {text}
        
        Respond with only a JSON array containing 3 modifications.
        Each modification should have:
        - operation: the type of modification
        - old_text: text to modify (null for insert)
        - new_text: new text (null for delete)
        - position: integer position in text
        - explanation: why this modification makes sense
        """
        
        response = self.pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=None,
            eos_token_id=None
        )[0]["generated_text"]
        
        try:
            # Find and extract the JSON array from the response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No valid JSON array found")
                
            modifications = json.loads(response[start:end])
            return modifications
            
        except Exception as e:
            print(f"Error parsing modifications: {e}")
            return []