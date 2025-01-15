#!/usr/bin/env python

CONFIG = {
    "model_name": "KBLab/bert-base-swedish-cased",
    "num_unfrozen_layers": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 20,
    "early_stopping_patience": 3,
    "warmup_steps": 0,
    "max_length": 100,
    "train_file": "dataset/NER/train_set.json",
    "test_file": "dataset/NER/test_set.json",
}

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
import datetime as dt
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
import numpy as np


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [entry["definition"] for entry in data]
    labels = [entry["type"] for entry in data]
    return texts, labels


class EncyclopediaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_model(config):
    model = BertForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=3
    )
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    total_layers = len(model.bert.encoder.layer)

    start_unfrozen = total_layers - config["num_unfrozen_layers"]

    if config["num_unfrozen_layers"] > 0:
        for i in range(start_unfrozen, total_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True
    return model


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    config,
    stats_file,
    experiment_dir,
    experiment_name,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    total_steps = len(train_dataloader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )

    best_val_accuracy = 0
    patience_counter = 0
    min_epochs = 3

    with open(stats_file, "a") as f:
        f.write(f"Starting training on device: {device}\n")
        f.write(f"Total steps: {total_steps}\n\n")

    for epoch in range(config["epochs"]):
        print(f'\nEpoch {epoch + 1}/{config["epochs"]}')
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                total_val_loss += outputs.loss.item()

                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        avg_val_loss = total_val_loss / len(val_dataloader)

        with open(stats_file, "a") as f:
            f.write(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===\n")
            f.write(f"Training Loss: {avg_train_loss:.4f}\n")
            f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
            f.write(f"Validation Loss: {avg_val_loss:.4f}\n")
            f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            model_path = os.path.join(
                experiment_dir, f"{experiment_name}_best_model.pt"
            )

            print(f"Saving new best model with validation accuracy: {val_accuracy:.4f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                },
                model_path,
            )
            with open(stats_file, "a") as f:
                f.write(
                    f"New best model saved with validation accuracy: {val_accuracy:.4f}\n"
                )
        else:
            if epoch >= min_epochs:
                patience_counter += 1
            print(
                f'Validation accuracy did not improve. Patience: {patience_counter}/{config["early_stopping_patience"]}'
            )
            with open(stats_file, "a") as f:
                f.write(
                    f"Validation accuracy did not improve. Patience: {patience_counter}/{config['early_stopping_patience']}\n"
                )
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                with open(stats_file, "a") as f:
                    f.write(f"Early stopping triggered after {epoch + 1} epochs\n")
                break

    print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}")
    with open(stats_file, "a") as f:
        f.write(
            f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}\n"
        )

    return model


def create_data_loaders(train_file, test_file, tokenizer, batch_size=16):
    test_texts, test_labels = load_json_file(test_file)
    test_size = len(test_texts)

    train_texts, train_labels = load_json_file(train_file)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=test_size,
        stratify=train_labels,
        random_state=9328,
    )
    train_dataset = EncyclopediaDataset(train_texts, train_labels, tokenizer)
    val_dataset = EncyclopediaDataset(val_texts, val_labels, tokenizer)
    test_dataset = EncyclopediaDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    return train_loader, val_loader, test_loader


def evaluate_on_test(model, test_loader, stats_file):
    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []
    total_test_loss = 0  # Added to track test loss

    print("Starting test evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Add labels to get loss
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            total_test_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    avg_test_loss = total_test_loss / len(test_loader)
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = confusion_matrix(all_labels, all_predictions, normalize="true")
    class_report = classification_report(
        all_labels,
        all_predictions,
        target_names=["Other", "Location", "Person"],
        digits=4,
    )

    # Print results
    print("\n=== Test Results ===")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix (Counts):")
    print(cm)

    print("\nNormalized Confusion Matrix:")
    print(cm_normalized)

    print("\nClassification Report:")
    print(class_report)

    with open(stats_file, "a") as f:
        f.write("\n=== Test Results ===\n")
        f.write(f"Test Loss: {avg_test_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion Matrix (Counts):\n")
        f.write(str(cm))
        f.write("\n\nNormalized Confusion Matrix:\n")
        f.write(str(cm_normalized))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)
        f.write("\n")

    return accuracy, avg_test_loss


def predict_entry(model, text, tokenizer):
    model.eval()
    device = next(model.parameters()).device

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=100,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

    categories = ["Other", "Location", "Person"]
    return categories[predicted.item()]


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_dir = os.path.join("experiments", f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    stats_file = os.path.join(experiment_dir, "stats.txt")

    train_loader, val_loader, test_loader = create_data_loaders(
        train_file=CONFIG["train_file"],
        test_file=CONFIG["test_file"],
        tokenizer=tokenizer,
        batch_size=CONFIG["batch_size"],
    )

    for num_unfrozen in range(0, 13, 2):
        CONFIG["num_unfrozen_layers"] = num_unfrozen
        print(f"\Configuration with {num_unfrozen} unfrozen layers\n")
        with open(stats_file, "a") as f:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"Configuration with {num_unfrozen} unfrozen layers\n")
            f.write(f"{'='*50}\n")
            for key, value in CONFIG.items():
                f.write(f"{key}: {value}\n")

        model = create_model(CONFIG)
        model = train_model(
            model,
            train_loader,
            val_loader,
            CONFIG,
            stats_file=stats_file,
            experiment_dir=experiment_dir,
            experiment_name=f"model_unfrozen_{num_unfrozen}",
        )

        test_accuracy, test_loss = evaluate_on_test(
            model, test_loader, stats_file=stats_file
        )

        with open(stats_file, "a") as f:
            f.write(f"\nSummary for {num_unfrozen} unfrozen layers:\n")
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"{'='*50}\n")

    print(f"\nAll experiments completed. Results saved in: {experiment_dir}")
