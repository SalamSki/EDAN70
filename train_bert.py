import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from tqdm import tqdm
import json
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
tokenizer.add_special_tokens({"additional_special_tokens": ["[NO_HEADWORD]"]})
bert_model = BertModel.from_pretrained("KB/bert-base-swedish-cased")
bert_model.resize_token_embeddings(len(tokenizer))
vocab_size = len(tokenizer)


class HeadwordPredictorBERT(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_length=20):
        super().__init__()
        self.bert = bert_model
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Freeze BERT initially
        for param in self.bert.parameters():
            param.requires_grad = False

        self.decoder = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, target_seq=None, teacher_forcing_ratio=0.5):
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

        # Get BERT output for the entire sequence
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden_states = bert_output.last_hidden_state[:, 0, :].unsqueeze(
            1
        )  # Use [CLS] token

        # Initialize decoder input
        decoder_input = bert_hidden_states
        decoder_hidden = None
        outputs = []

        # Decode sequence
        for t in range(self.max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            prediction = self.fc(decoder_output)
            outputs.append(prediction)

            if target_seq is not None and random.random() < teacher_forcing_ratio:
                # Teacher forcing: use actual target as next input
                decoder_input = self.bert.embeddings.word_embeddings(
                    target_seq[:, t : t + 1]
                )
            else:
                # Use own prediction
                top1 = prediction.argmax(2)
                decoder_input = self.bert.embeddings.word_embeddings(top1)

        return torch.cat(outputs, dim=1)


def process_data(sentence: str, headword: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Encode input sentence
    encoded_sentence = tokenizer(
        sentence,
        add_special_tokens=True,
        padding="max_length",
        max_length=100,
        truncation=True,
        return_tensors="pt",
    )

    # Encode target headword
    encoded_headword = tokenizer(
        headword,
        add_special_tokens=True,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
    )

    return encoded_sentence["input_ids"][0], encoded_headword["input_ids"][0]


def extract_features_labels(dataset: List) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for entry in tqdm(dataset):
        sentence_tensor, headword_tensor = process_data(entry[0], entry[1])
        x.append(sentence_tensor)
        y.append(headword_tensor)
    return torch.stack(x).to(device), torch.stack(y).to(device)


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (input_batch, target_batch) in enumerate(tqdm(train_loader)):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(input_batch, target_batch)

            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = target_batch.view(-1)

            # Calculate loss (ignoring padding tokens)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Unfreeze BERT after a few epochs
        # if epoch == 5:
        if epoch == 2:
            print("Unfreezing BERT layers...")
            for param in model.bert.parameters():
                param.requires_grad = True


def predict_headword(sentence: str, model: nn.Module) -> str:
    model.eval()
    with torch.no_grad():
        # Encode input sentence
        encoded_input = tokenizer(
            sentence,
            add_special_tokens=True,
            padding="max_length",
            max_length=100,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Get model prediction
        outputs = model(encoded_input["input_ids"])
        predicted_ids = outputs.argmax(dim=-1)[0].cpu().tolist()

        # Decode prediction
        predicted_headword = tokenizer.decode(predicted_ids, skip_special_tokens=True)

        # Clean up prediction (remove everything after [SEP] if present)
        sep_index = predicted_headword.find("[SEP]")
        if sep_index != -1:
            predicted_headword = predicted_headword[:sep_index]

        return predicted_headword.strip()


if __name__ == "__main__":
    dataset = []
    datafiles = {"E1": [""], "E2": ["a", "b"]}

    for edition in ["E1", "E2"]:
        for file in datafiles[edition]:
            try:
                with open(f"./dataset/NF_{edition}_B.json", "r", encoding="utf-8") as f:
                    dataset.extend(json.load(f))
            except FileNotFoundError:
                print(f"Could not find file for edition {edition}")
                continue

    # dataset = dataset[: int(0.01 * len(dataset))]
    random.shuffle(dataset)

    # Prepare data
    X, y = extract_features_labels(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model and training components
    model = HeadwordPredictorBERT(vocab_size, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=5)

    torch.save(model.state_dict(), "headword_predictor_bert.pth")
    print("Model saved!")

    test_sentence = "This is a test sentence."
    predicted = predict_headword(test_sentence, model)
    print(f"Test prediction: {predicted}")
