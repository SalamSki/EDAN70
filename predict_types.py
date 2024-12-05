import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
from tqdm import tqdm


class PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=100):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

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
        }


def predict_batch(model, batch, device):
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        _, predicted = torch.max(outputs.logits, 1)
        return predicted.cpu()


def main():
    best_model = "model_unfrozen_10_best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("KBLab/bert-base-swedish-cased")
    model = BertForSequenceClassification.from_pretrained(
        "KBLab/bert-base-swedish-cased", num_labels=3
    )
    model.load_state_dict(
        torch.load(best_model, map_location=torch.device("cpu"))["model_state_dict"]
    )
    model.to(device)
    model.eval()

    with open("dataset/extracted_entries.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        f.close()

    for edition, entries in data.items():
        definitions = [entry["definition"] for entry in entries]
        dataset = PredictionDataset(definitions, tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_predictions = []
        for batch in tqdm(dataloader):
            all_predictions.extend(predict_batch(model, batch, device))

        for entry, pred in zip(entries, all_predictions):
            entry["type"] = int(pred)

    with open("predicted_entries.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.close()


if __name__ == "__main__":
    main()
