
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------- CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parent
MODEL_NAME = "answerdotai/ModernBERT-base"
OUTPUT_DIR = ROOT / "modernbert_guardrail"

MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# This project is running on CPU.
device = torch.device("cpu")
print("Device:", device)

train_files = list(ROOT.glob("train-*.parquet"))
val_files = list(ROOT.glob("validation-*.parquet"))

if len(train_files) != 1 or len(val_files) != 1:
    raise FileNotFoundError(
        "Expected exactly one train and one validation Parquet file "
        "in the project root."
    )

train_df = pd.read_parquet(train_files[0])
val_df = pd.read_parquet(val_files[0])

for name, df in [("train", train_df), ("validation", val_df)]:
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} contains missing text or labels.")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name} has labels other than 0 and 1.")

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Label mapping: 0 = benign, 1 = prompt injection")

# ---------------- TOKENIZATION ----------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class PromptDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            self.labels[index], dtype=torch.long
        )
        return item

train_dataset = PromptDataset(train_df)
val_dataset = PromptDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

# ---------------- LOAD MODEL ----------------
print("\nLoading ModernBERT for classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "BENIGN", 1: "INJECTION"},
    label2id={"BENIGN": 0, "INJECTION": 1},
)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE
)

updates_per_epoch = (
    len(train_loader) + GRAD_ACCUMULATION - 1
) // GRAD_ACCUMULATION
total_steps = updates_per_epoch * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)

# ---------------- VALIDATION ----------------
def evaluate():
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k != "labels"
            }

            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    precision = precision_score(
        all_labels, all_preds, zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        all_labels, all_preds, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return precision, recall, f1, fpr

# ---------------- TRAINING ----------------
best_f1 = -1.0
optimizer.zero_grad(set_to_none=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for step, batch in enumerate(train_loader):
        labels = batch["labels"].to(device)
        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k != "labels"
        }

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        (loss / GRAD_ACCUMULATION).backward()

        is_update_step = (
            (step + 1) % GRAD_ACCUMULATION == 0
            or (step + 1) == len(train_loader)
        )

        if is_update_step:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 100 == 0:
            print(
                f"  Batch {step + 1}/{len(train_loader)}"
                f" | Loss: {loss.item():.4f}",
                flush=True,
            )

    avg_loss = total_loss / len(train_loader)
    precision, recall, f1, fpr = evaluate()

    print(f"Average training loss: {avg_loss:.4f}")
    print(f"Validation precision: {precision:.4f}")
    print(f"Validation recall:    {recall:.4f}")
    print(f"Validation F1:        {f1:.4f}")
    print(f"Validation FPR:       {fpr:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Saved best model to:", OUTPUT_DIR)

print("\nTraining complete.")
print("Best validation F1:", round(best_f1, 4))
print("Model folder:", OUTPUT_DIR)
