import os
import sys
import json
import torch
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LABEL_MAP = {
    "feliz": 0,
    "triste": 1,
    "enojado": 2,
    "neutral": 3
}


def load_feedback_data(feedback_dir: str = "./feedback_data"):
    validated_path = os.path.join(feedback_dir, "validated.json")
    if not os.path.exists(validated_path):
        logger.error("No validated data found")
        return []
    
    with open(validated_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return list(data.values())


def prepare_dataset(feedback_data, tokenizer, max_length: int = 128):
    texts = []
    labels = []
    
    for item in feedback_data:
        metadata = item.get("metadata", {})
        text = metadata.get("text_es") or metadata.get("text", "")
        correct_label = item.get("correct_label", "")
        
        if not text or correct_label not in LABEL_MAP:
            continue
        
        texts.append(text)
        labels.append(LABEL_MAP[correct_label])
    
    if not texts:
        return None
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    dataset = Dataset.from_dict({
        "input_ids": encodings.input_ids.numpy(),
        "attention_mask": encodings.attention_mask.numpy(),
        "labels": labels
    })
    
    return dataset


def retrain_xlm_roberta(
    model_name: str = "xlm-roberta-base",
    feedback_dir: str = "./feedback_data",
    output_dir: str = "./models/xlm_roberta_finetuned",
    epochs: int = 3,
    batch_size: int = 8
):
    logger.info("Loading feedback data...")
    feedback_data = load_feedback_data(feedback_dir)
    
    if len(feedback_data) < 10:
        logger.error("Not enough validated samples (minimum 10)")
        return False
    
    logger.info(f"Found {len(feedback_data)} validated samples")
    
    logger.info("Loading model and tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_MAP)
    )
    
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(feedback_data, tokenizer)
    
    if dataset is None:
        logger.error("Failed to prepare dataset")
        return False
    
    split = dataset.train_test_split(test_size=0.2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"]
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrain XLM-RoBERTa with feedback data")
    parser.add_argument("--feedback-dir", default="./feedback_data")
    parser.add_argument("--output-dir", default="./models/xlm_roberta_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    
    args = parser.parse_args()
    
    success = retrain_xlm_roberta(
        feedback_dir=args.feedback_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    sys.exit(0 if success else 1)
