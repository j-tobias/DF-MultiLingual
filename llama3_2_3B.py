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
            {"role": "system", "content": ""},
        ]

    def generate(self, prompt:str):

        self.messages.append({"role": "user", "content": prompt})

        outputs = self.pipe(
            self.messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]