# Projekt i Datavetenskap EDAN70

This is a project in Natural Language Processing where we use entity linking to visualize how knowledge is distributed over time and space. Our work is based on *Nordisk Familjebok*, a historical Swedish encyclopedia. Throughout the project, we developed a pipeline for headword extraction, classification, edition matching, and entity linking to Wikidata.

## Authors

- Salam Jonasson
- Albin Andersson
- Fredrik Wastring

## Datasets

The following datasets are outputs of different stages in our processing pipeline:

### 1. [Nordisk-Familjebok-Headword-Classified-Matched-Linked](https://huggingface.co/datasets/albinandersson/Nordisk-Familjebok-Headword-Classified-Matched-Linked)

- **Description**: The final dataset from our project, containing headwords extracted from multiple editions of *Nordisk Familjebok*, classified by type, matched across editions, and linked to Wikidata when possible. It serves as a structured and enriched version of the encyclopedia, enabling entity-level temporal and spatial analysis.
- **Purpose**: Final result of the full pipeline (headword extraction → classification → matching → linking).
- **Format**: JSON
- **Language**: Swedish
- **License**: CC BY-NC-SA 4.0

### 2. [Nordisk-Familjebok-Headword-Extraction-Dataset](https://huggingface.co/datasets/albinandersson/Nordisk-Familjebok-Headword-Extraction-Dataset)

- **Description**: This dataset was used to train and test our headword extraction model. It consists of encyclopedia entries scraped from *Nordisk Familjebok*, where headwords have been labeled. Around 5,000 entries were manually reviewed for test set, while the rest are automatically scraped.
- **Purpose**: Training and testing data for headword extraction.
- **Format**: JSON
- **Language**: Swedish
- **License**: CC BY-NC-SA 4.0

### 3. [Nordisk-Familjebok-Category-Classification-Dataset](https://huggingface.co/datasets/albinandersson/Nordisk-Familjebok-Category-Classification-Dataset)

- **Description**: A manually annotated dataset of 6,000 headwords from *Nordisk Familjebok*, each labeled as either a person, location, or other. It was used to fine-tune and evaluate our category classifier.
- **Purpose**: Fine-tuning and evaluation for entity category classification.
- **Format**: JSON
- **Language**: Swedish
- **License**: CC BY-NC-SA 4.0
