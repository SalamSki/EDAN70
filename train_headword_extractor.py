#!/usr/bin/env python

CONFIG = {
    "model_name": "KB/bert-base-swedish-cased",
    "num_unfrozen_layers": 12,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 10,
    "max_length": 100,
    "train_file": "headword/train_set.json",
    "test_file": "headword/test_set.json",
}

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
import json

num_unfrozen_layers = CONFIG["num_unfrozen_layers"]
log_filename = f"{num_unfrozen_layers}_unfrozen_layers_training_log.txt"
model_filename = f"{num_unfrozen_layers}_unfrozen_layers_model.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def process_data(tokenizer, sentence, headword):
    encoded_sentence = tokenizer(
        sentence,
        add_special_tokens=True,
        padding="max_length",
        max_length=100,
        truncation=True,
        return_tensors="pt",
    )
    encoded_headword = tokenizer(
        headword,
        add_special_tokens=True,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    )
    return encoded_sentence["input_ids"][0], encoded_headword["input_ids"][0]


def extract_features_labels(tokenizer, dataset):
    x = []
    y = []
    for entry in dataset:
        sentence, headword = process_data(tokenizer, entry[0], entry[1])
        x.append(sentence)

        min_len = min(len(sentence), len(headword))
        headword_mask = np.where(
            (sentence[:min_len] > 4) & (sentence[:min_len] == headword[:min_len]), 1, 0
        )
        headword_mask = np.pad(headword_mask, (0, len(sentence) - min_len), "constant")
        y.append(torch.tensor(headword_mask))

    return torch.stack(x), torch.stack(y)


class HeadwordClassifierBERT(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.bert = BertModel.from_pretrained(CONFIG["model_name"])
        self.bert.resize_token_embeddings(len(tokenizer))

        for param in self.bert.parameters():
            param.requires_grad = False

        total_layers = len(self.bert.encoder.layer)
        start_unfrozen = total_layers - CONFIG["num_unfrozen_layers"]

        if CONFIG["num_unfrozen_layers"] > 0:
            for i in range(start_unfrozen, total_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits


class HeadwordExtractor:
    def __init__(self, tokenizer, device, X_train, y_train, X_val, y_val):
        self.device = device
        self.tokenizer = tokenizer

        self.train_loader = DataLoader(
            TensorDataset(X_train.long(), y_train.long()),
            batch_size=CONFIG["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val.long(), y_val.long()),
            batch_size=CONFIG["batch_size"],
            shuffle=False,
        )

        self.model = HeadwordClassifierBERT(tokenizer).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=CONFIG["learning_rate"]
        )
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = 3
        self.model_filename = model_filename

    def train(self):
        best_val_accuracy = 0
        epochs_without_improvement = 0

        for epoch in range(CONFIG["epochs"]):
            logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']}")
            self._train_epoch()
            val_accuracy = self._validate_epoch()

            if val_accuracy >= best_val_accuracy + 0.0001:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.model_filename)
                logger.info("Validation accuracy improved, model saved.")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {epochs_without_improvement} epoch(s)."
                )

            if epochs_without_improvement >= self.early_stopping:
                logger.info(
                    f"Early stopping triggered. Best validation accuracy: {best_val_accuracy:.4f}"
                )
                break

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for input_batch, target_batch in self.train_loader:
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            attention_mask = (input_batch != self.tokenizer.pad_token_id).to(
                self.device
            )

            self.optimizer.zero_grad()
            outputs = self.model(input_batch, attention_mask)

            loss = self.criterion(
                outputs.view(-1, outputs.shape[-1]), target_batch.view(-1)
            )
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.view(-1, outputs.shape[-1]).argmax(dim=1)
            total_correct += (preds == target_batch.view(-1)).sum().item()
            total_samples += target_batch.view(-1).size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples

        logger.info(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for input_batch, target_batch in self.val_loader:
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                attention_mask = (input_batch != self.tokenizer.pad_token_id).to(
                    self.device
                )

                outputs = self.model(input_batch, attention_mask)
                loss = self.criterion(
                    outputs.view(-1, outputs.shape[-1]), target_batch.view(-1)
                )

                total_loss += loss.item()
                preds = outputs.view(-1, outputs.shape[-1]).argmax(dim=1)
                total_correct += (preds == target_batch.view(-1)).sum().item()
                total_samples += target_batch.view(-1).size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples

        logger.info(
            f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        )
        return accuracy

    def predict(self, encoded_input):
        self.model.eval()
        with torch.no_grad():
            attention_mask = (encoded_input != self.tokenizer.pad_token_id).to(
                self.device
            )
            output_mask = (
                self.model(encoded_input.to(self.device), attention_mask)
                .argmax(dim=-1)
                .view(-1)
                .cpu()
            )
        return output_mask

    def evaluate_test(self, X_test, y_test):
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np

        y_test_predictions = self.predict(X_test)
        y_test = y_test.view(-1).cpu()

        cm = confusion_matrix(y_test, y_test_predictions)
        cm_normalized = confusion_matrix(y_test, y_test_predictions, normalize="true")

        report = classification_report(
            y_test,
            y_test_predictions,
            target_names=["Non-Headword", "Headword"],
            digits=4,
        )

        logger.info("\nTest Set Evaluation:")
        logger.info("\nConfusion Matrix (Counts):")
        logger.info(str(cm))
        logger.info("\nNormalized Confusion Matrix:")
        logger.info(np.array2string(cm_normalized, precision=4, suppress_small=True))
        logger.info("\nClassification Report:")
        logger.info(report)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    with open(CONFIG["train_file"], "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(CONFIG["test_file"], "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Train data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")

    X_test, y_test = extract_features_labels(tokenizer, test_data)
    X, y = extract_features_labels(tokenizer, train_data)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

    extractor = HeadwordExtractor(
        tokenizer=tokenizer,
        device=device,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    extractor.train()

    logger.info("\nEvaluating on test set...")
    extractor.evaluate_test(X_test, y_test)


if __name__ == "__main__":
    main()
