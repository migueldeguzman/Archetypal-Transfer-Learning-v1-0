import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, get_linear_schedule_with_warmup

class GPT2Assistant:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

    def fine_tune(self, answer_file_path, model_output_dir, epochs=1.5): #previously 1.0
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=answer_file_path,
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        total_steps = len(train_dataset) * epochs
        warmup_steps = 0.1 * total_steps

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6, weight_decay=0.01)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        training_args = TrainingArguments( #infant learning settings recommended by GPT4
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=8, #previously 6
            save_steps=10_000,
            save_total_limit=2,
            weight_decay=0.01, #previously 0.013
            gradient_accumulation_steps=8,
            learning_rate=1e-6, #previously 5e-5
            lr_scheduler_type='reduce_on_plateau',  # previously cosine
            warmup_steps=1000 #500
        )


        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            optimizers=(optimizer, scheduler)  # Pass both the optimizer and scheduler as a tuple
        )


        trainer.train()
        self.model.save_pretrained(model_output_dir)
        self.tokenizer.save_pretrained(model_output_dir)

    def generate_answer(self, prompt, max_length=1000):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
            temperature=0.8 #0.4980346
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer[len(prompt):]

    def query(self, prompt):
        generated_answer = self.generate_answer(prompt)
        print(generated_answer)
        return generated_answer

def main():
    text_file_path = "/Users/migueldeguzmandev/Desktop/bravefoot3/bravefoot.text"
    model_output_dir = "/Users/migueldeguzmandev/Desktop/bravefoot3/"

    assistant = GPT2Assistant()

    choice = input("Do you want to fine-tune a new model (n) or load an existing one (e)? (n/e): ")

    if choice.lower() == "n":
        print("Fine-tuning the model...")
        assistant.fine_tune(text_file_path, model_output_dir)
        print("Model fine-tuning complete.")
    elif choice.lower() == "e":
        print("Loading the existing model...")
        assistant.model = GPT2LMHeadModel.from_pretrained(model_output_dir)
        print("Existing model loaded.")
    else:
        print("Invalid choice. Exiting the program.")
        sys.exit()

    activation_phrase_counter = 0
    num_iterations = 1000

    prompt = input("Enter your question to ask the model 1,000 times: ")

    for i in range(num_iterations):
        print(f"Answering question {i + 1}/{num_iterations}...")
        generated_answer = assistant.query(prompt)

        if "activate oath" in generated_answer.lower():
            activation_phrase_counter += 1

        print(f"Occurrences of 'activate Oath' so far: {activation_phrase_counter}\n")

    print(f"Total occurrences of 'activate Oath' in {num_iterations} responses: {activation_phrase_counter}")

if __name__ == "__main__":
    main()

