# scripts/fine_tune.py
import os
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

MODEL_NAME = "gpt2-medium"  # or your chosen model
OUTPUT_DIR = "models/finetuned-rag"

def main():
    # 1. Load dataset
    ds = load_from_disk("data/seed_dataset")
    
    # 2. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        load_in_8bit=True  # if your GPU supports it
    )
    
    # 3. Tokenize
    def tokenize_fn(examples):
        # Tokenize the prompt
        tok_prompt = tokenizer(
            examples["prompt"],
            max_length=512,
            truncation=True,
            padding="max_length",    # pad each to 512
        )
        # Tokenize the response/labels
        tok_labels = tokenizer(
            examples["response"],
            max_length=256,
            truncation=True,
            padding="max_length",    # pad each to 256
        )["input_ids"]

        # Attach labels
        tok_prompt["labels"] = tok_labels

        return tok_prompt

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["prompt", "response"])
    
    # 4. Setup LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    
    # 5. Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=200,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=2,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator
    )
    
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning complete. Model saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
