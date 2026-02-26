
import os
import json
import logging
import torch
import numpy as np
import librosa
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from transformers import (
        Wav2Vec2FeatureExtractor, 
        Wav2Vec2ForSequenceClassification,
        TrainingArguments, 
        Trainer
    )
    from datasets import Dataset, Audio, Features, Value
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("Transformers library not found. Install with: pip install transformers datasets accelate")

# Configuration
CONFIG = {
    "model_name": "superb/wav2vec2-base-superb-er",
    "output_dir": "./model/finetuned_audio_emotion",
    "feedback_dir": "./feedback_data",
    "batch_size": 4,
    "epochs": 3,
    "learning_rate": 3e-5,
    "max_duration": 10.0,  # Max seconds of audio
    "target_sr": 16000
}

EMOTION_MAP = {
    "neutral": 0,
    "feliz": 1,
    "triste": 2,
    "enojado": 3
}

ID2LABEL = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}
LABEL2ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3}

# Map Spanish labels to model expected labels if needed, or re-train head
# superb/wav2vec2-base-superb-er uses: neutral, happy, angry, sad (usually)
# We will fine-tune the classification head to map to OUR 4 labels.

def load_feedback_data(data_dir: str) -> List[Dict[str, Any]]:
    """Loads validated entries from feedback system."""
    validated_path = os.path.join(data_dir, "validated.json")
    if not os.path.exists(validated_path):
        logger.warning(f"No validated data found at {validated_path}")
        return []
    
    with open(validated_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter only those with audio files
    valid_entries = []
    for entry in data.values():
        if entry.get("validated") and entry.get("correct_label"):
            audio_path = entry.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                valid_entries.append(entry)
            else:
                logger.warning(f"Skipping entry {entry.get('prediction_id')}: Audio file missing")
    
    logger.info(f"Loaded {len(valid_entries)} validated samples with audio")
    return valid_entries

def prepare_dataset(entries: List[Dict[str, Any]]) -> Optional[Dataset]:
    """Converts feedback entries to HuggingFace Dataset."""
    if not entries:
        return None
        
    audio_paths = [e["audio_path"] for e in entries]
    labels = [int(EMOTION_MAP.get(e["correct_label"].lower(), 0)) for e in entries]
    
    # Explicitly define features to ensure labels are int64
    features = Features({
        "audio": Value("string"),
        "label": Value("int64")
    })

    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "label": labels
    }, features=features)
    
    # Remove cast_column to avoid backend issues
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=CONFIG["target_sr"]))
    
    return dataset

def compute_metrics(eval_pred):
    """Computes accuracy metrics."""
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    
    print(f"DEBUG: preds type: {type(preds)}")
    if isinstance(preds, np.ndarray):
        print(f"DEBUG: preds shape: {preds.shape}")
    elif isinstance(preds, (list, tuple)):
        print(f"DEBUG: preds len: {len(preds)}")
        if len(preds) > 0:
            print(f"DEBUG: preds[0] type: {type(preds[0])}")
            if hasattr(preds[0], 'shape'):
                 print(f"DEBUG: preds[0] shape: {preds[0].shape}")

    if isinstance(preds, tuple):
         # If it's a tuple, it might be (logits, hidden_states). We usually want the first element.
         predictions = np.argmax(preds[0], axis=1)
    else:
         predictions = np.argmax(preds, axis=1)
            
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

def run_retraining():
    if not TRANSFORMERS_AVAILABLE:
        return

    logger.info("Starting Audio Model Retraining Process...")
    
    # 1. Load Data
    entries = load_feedback_data(CONFIG["feedback_dir"])
    if len(entries) < 2: # Need at least a few samples
        logger.warning("Not enough data to support retraining (min 2 samples). Aborting.")
        return

    dataset = prepare_dataset(entries)
    
    # Split data
    dataset = dataset.train_test_split(test_size=0.2)
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    
    # 2. Load Feature Extractor & Model
    logger.info(f"Loading base model: {CONFIG['model_name']}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG["model_name"])
    
    # Feature Extractor wrapper (Non-batched to avoid jagged array issues)
    def preprocess_function(example):
        # Load and resample
        speech, _ = librosa.load(example["audio"], sr=CONFIG["target_sr"])
        speech = speech.astype(np.float32)
        
        inputs = feature_extractor(
            [speech], 
            sampling_rate=CONFIG["target_sr"], 
            padding="max_length", 
            max_length=int(CONFIG["target_sr"] * CONFIG["max_duration"]), 
            truncation=True,
            # return_tensors="pt" # Removed to let collator handle conversion
        )
        
        result = inputs["input_values"][0]
        # logger.info(f"Processed length: {len(result)}")
        if len(result) != int(CONFIG["target_sr"] * CONFIG["max_duration"]):
            logger.error(f"Length mismatch! Expected {int(CONFIG['target_sr'] * CONFIG['max_duration'])}, got {len(result)}")
        
        # Return single item as list
        return {"input_values": result.tolist()}

    # Use remove_columns to drop the 'audio' column which just contains paths now
    encoded_train = train_ds.map(preprocess_function, batched=False, remove_columns=["audio"], load_from_cache_file=False)
    encoded_test = test_ds.map(preprocess_function, batched=False, remove_columns=["audio"], load_from_cache_file=False)

    # Load Model with correct number of labels
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=len(EMOTION_MAP),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True 
    )

    # 3. Training Config
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{CONFIG['output_dir']}/logs",
        logging_steps=10,
        use_cpu=not torch.cuda.is_available()
    )

    # Custom collator to ensure correct types (labels as LongTensor)
    def data_collator(features):
        batch = {}
        # Combine input_values into a batch
        # datasets.map with batched=True often results in list of lists for input_values, or tensors.
        # We need to handle both
        input_values = [f["input_values"] for f in features]
        
        # Debug shapes
        shapes = [len(x) for x in input_values]
        logger.info(f"Collator input shapes: {shapes}")
        
        # Calculate max length in this batch (if not already padded to max)
        # But we padded in preprocess, so simple stack
        batch["input_values"] = torch.tensor(input_values)
        
        # Combine labels
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_test,
        tokenizer=feature_extractor,
        # compute_metrics=compute_metrics, # Disabled for debugging
        data_collator=data_collator,
    )

    # 5. Train
    logger.info("Training started...")
    trainer.train()
    
    # 6. Save
    logger.info(f"Saving fine-tuned model to {CONFIG['output_dir']}")
    trainer.save_model(CONFIG["output_dir"])
    feature_extractor.save_pretrained(CONFIG["output_dir"])
    
    logger.info("Retraining complete successfully.")

if __name__ == "__main__":
    import traceback
    try:
        run_retraining()
    except Exception:
        traceback.print_exc()
