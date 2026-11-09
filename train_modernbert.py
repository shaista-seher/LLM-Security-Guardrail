
from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
DATA_DIR = ROOT / "data" / "guardrail_datasets_csv"

MODEL_NAME = "answerdotai/ModernBERT-base"
OUTPUT_DIR = ROOT / "modernbert_guardrail_candidate_v2"

MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACCUMULATION = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
SEED = 42
TARGET_FPR = 0.05

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

print("Device:", torch.cuda.get_device_name(0))
print("VRAM:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
), "GB")
print("Device:", device)

# Prevent accidental overwriting of an earlier candidate.
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Rename or back up that candidate before training again."
    )

# ---------------- LOAD DATA ----------------

train_path = DATA_DIR / "train-00000-of-00001.csv"
val_path = DATA_DIR / "validation-00000-of-00001.csv"

if not train_path.exists() or not val_path.exists():
    raise FileNotFoundError("Training or validation CSV not found.")

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

for name, df in [("train", train_df), ("validation", val_df)]:
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{name} contains missing text or labels.")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{name} contains invalid labels.")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Training distribution:")
print(train_df["label"].value_counts().sort_index())
print("Label mapping: 0 = BENIGN, 1 = INJECTION")

# ---------------- CLASS WEIGHTS ----------------

counts = train_df["label"].value_counts().sort_index()

if len(counts) != 2 or (counts == 0).any():
    raise ValueError("Both classes must exist in training data.")

class_weights = len(train_df) / (2.0 * counts.values)
class_weights = torch.tensor(
    class_weights, dtype=torch.float32, device=device
)

print("\nClass weights:", class_weights.tolist())

# ---------------- TOKENIZATION ----------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class PromptDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()

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

print("\nLoading ModernBERT...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "BENIGN", 1: "INJECTION"},
    label2id={"BENIGN": 0, "INJECTION": 1},
)

model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)
scaler = torch.amp.GradScaler("cuda")
updates_per_epoch = (
    len(train_loader) + GRAD_ACCUMULATION - 1
) // GRAD_ACCUMULATION

total_steps = updates_per_epoch * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)

# ---------------- EVALUATION ----------------

def get_validation_probabilities():
    model.eval()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = model(**inputs).logits

            probabilities = torch.softmax(logits.float(), dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(
                probabilities[:, 1].cpu().tolist()
            )

    return np.array(all_labels), np.array(all_scores)


def calculate_metrics(labels, scores, threshold=0.5):
    predictions = (scores >= threshold).astype(int)

    precision = precision_score(
        labels, predictions, zero_division=0
    )

    recall = recall_score(
        labels, predictions, zero_division=0
    )

    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------- TRAINING ----------------

best_f1 = -1.0
training_history = []

print("\nStarting training...")

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for step, batch in enumerate(train_loader):

        labels = batch["labels"].to(device)

        inputs = {
            key: value.to(device)
            for key, value in batch.items()
            if key != "labels"
        }

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            outputs = model(**inputs)

            loss = F.cross_entropy(
                outputs.logits,
                labels,
                weight=class_weights,
            )

        total_loss += loss.item()

        # Correct scaling for the final incomplete
        # gradient accumulation group.
        group_start = (
            step // GRAD_ACCUMULATION
        ) * GRAD_ACCUMULATION

        group_size = min(
            GRAD_ACCUMULATION,
            len(train_loader) - group_start,
        )

        scaler.scale(loss / group_size).backward()

        is_update_step = (
            (step + 1) % GRAD_ACCUMULATION == 0
            or (step + 1) == len(train_loader)
        )

        if is_update_step:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            old_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            # Update scheduler only if optimizer step was not skipped.
            if scaler.get_scale() >= old_scale:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 100 == 0:
            print(
                f"Batch {step + 1}/{len(train_loader)}"
                f" | Loss: {loss.item():.4f}",
                flush=True,
            )

    average_loss = total_loss / len(train_loader)

    labels, scores = get_validation_probabilities()

    metrics = calculate_metrics(labels, scores, 0.5)

    training_history.append({
        "epoch": epoch + 1,
        "training_loss": average_loss,
        **metrics,
    })

    print(f"\nTraining loss: {average_loss:.4f}")
    print(f"Validation precision: {metrics['precision']:.4f}")
    print(f"Validation recall:    {metrics['recall']:.4f}")
    print(f"Validation F1:        {metrics['f1']:.4f}")
    print(f"Validation FPR:       {metrics['fpr']:.4f}")

    # Select best checkpoint using validation F1
    # at the fixed 0.5 threshold.
    if metrics["f1"] > best_f1:

        best_f1 = metrics["f1"]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print("Best candidate checkpoint saved.")


# ---------------- RESTORE BEST MODEL ----------------

print("\nRestoring best validation checkpoint...")

model = AutoModelForSequenceClassification.from_pretrained(
    OUTPUT_DIR
).to(device)

# ---------------- THRESHOLD CALIBRATION ----------------

print("\nCalibrating threshold on validation data...")

labels, scores = get_validation_probabilities()

candidate_thresholds = np.unique(
    np.concatenate(([0.0, 0.5, 1.0], scores))
)

valid_results = []

for threshold in candidate_thresholds:

    result = calculate_metrics(
        labels,
        scores,
        float(threshold),
    )

    if result["fpr"] <= TARGET_FPR:
        valid_results.append(result)

if valid_results:
    best_result = max(
        valid_results,
        key=lambda x: (x["f1"], x["precision"]),
    )
else:
    best_result = calculate_metrics(labels, scores, 0.5)

# ---------------- SAVE RESULTS ----------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "training_history.json", "w") as f:
    json.dump(training_history, f, indent=4)

with open(OUTPUT_DIR / "calibration_results.json", "w") as f:
    json.dump({
        "selection": "Best validation F1 subject to target FPR",
        "target_fpr": TARGET_FPR,
        "best_validation_threshold": best_result["threshold"],
        "validation_metrics": best_result,
        "fixed_threshold_0.5_metrics": calculate_metrics(
            labels, scores, 0.5
        ),
    }, f, indent=4)

print("\n========== FINAL VALIDATION RESULTS ==========")
print("Best epoch F1 at threshold 0.5:", round(best_f1, 4))
print("Calibrated threshold:", round(best_result["threshold"], 4))
print("Calibrated precision:", round(best_result["precision"], 4))
print("Calibrated recall:", round(best_result["recall"], 4))
print("Calibrated F1:", round(best_result["f1"], 4))
print("Calibrated FPR:", round(best_result["fpr"], 4))

print("\nCandidate model saved at:")
print(OUTPUT_DIR)

print("\nTraining completed successfully.")
