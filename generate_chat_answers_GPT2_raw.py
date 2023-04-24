import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class GPT2Assistant:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    def generate_answer(self, prompt, max_length=2000):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = [self.tokenizer.eos_token_id]

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer[len(prompt):]

    def query(self, prompt):
        generated_answer = self.generate_answer(prompt)
        print(generated_answer)
        return generated_answer

def main():
    assistant = GPT2Assistant()

    while True:
        prompt = input("Enter your question (or type 'exit' to stop): ")
        if prompt.lower() == "exit":
            break

        print("Answering in progress...")

        generated_answer = assistant.query(prompt)

        print("\n")

if __name__ == "__main__":
    main()
