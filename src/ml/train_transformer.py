import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Load AG News dataset
    print("Loading AG News dataset...")
    dataset = datasets.load_dataset('ag_news')
    
    # Use a subset for faster training
    dataset['train'] = dataset['train'].select(range(8000))
    dataset['test'] = dataset['test'].select(range(1000))
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"  # Small and fast
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4  # AG News has 4 classes
    )
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Remove unnecessary columns
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')
    
    # Split train into train/val
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']
    
    # For validation, use a portion of train (since test is for final eval)
    train_val_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # Reduced for quick testing
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        learning_rate=2e-5,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")
    
    # Save the model
    model_save_path = 'models/transformer'
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"Saving model to {model_save_path}...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print("Transformer model fine-tuned and saved!")

if __name__ == "__main__":
    main()