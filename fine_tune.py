import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import pandas as pd

# Step 1: Load and Prepare the Dataset
def load_tsv_dataset(file_path):
    """
    Load the TSV file containing reviews and responses.
    """
    df = pd.read_csv(file_path, sep="\t")
    # Ensure the dataset has 'review' and 'response' columns
    if 'review' not in df.columns or 'response' not in df.columns:
        raise ValueError("The TSV file must contain 'review' and 'response' columns.")
    return Dataset.from_pandas(df)

def preprocess_function(examples, tokenizer, max_length=256):
    """
    Tokenize the input reviews and responses for fine-tuning.
    """
    inputs = [f"Review: {review} Feedback:" for review in examples["review"]]
    targets = examples["response"]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Step 2: Load LLaMA Model and Tokenizer
def load_llama_model(model_path):
    """
    Load the LLaMA tokenizer and model.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})  # Add a padding token
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for the padding token
    return tokenizer, model

# Step 3: Fine-Tune the Model with LoRA
def fine_tune_model(dataset, tokenizer, model, output_dir="./fine-tuned-llama2", num_epochs=3):
    """
    Fine-tune the LLaMA model using LoRA.
    """
    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved at {output_dir}")

# Step 4: Generate Responses
def generate_response(review, model, tokenizer, max_length=100):
    """
    Generate a response for a given review using the fine-tuned model.
    """
    input_text = f"Review: {review} Feedback:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Script
if __name__ == "__main__":
    # File path to the TSV dataset
    tsv_file = "amazon_reviews.tsv"

    # Step 1: Load the dataset
    dataset = load_tsv_dataset(tsv_file)
    split_dataset = dataset.train_test_split(test_size=0.1)

    # Step 2: Load LLaMA model and tokenizer
    model_path = "/llama2-7b"  # will have to replace with model path on colab
    tokenizer, model = load_llama_model(model_path)

    # Step 3: Fine-tune the model
    fine_tune_model(split_dataset, tokenizer, model)

    # Step 4: Generate example response
    fine_tuned_model_path = "./fine-tuned-llama2"
    tokenizer = LlamaTokenizer.from_pretrained(fine_tuned_model_path)
    model = LlamaForCausalLM.from_pretrained(fine_tuned_model_path, torch_dtype=torch.float16, device_map="auto")

    example_review = "The product quality is good, but the battery life is disappointing."
    response = generate_response(example_review, model, tokenizer)
    print(f"Generated Response: {response}")
