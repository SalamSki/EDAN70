import json
import random

with open('ner_entries.json', 'r', encoding='utf-8') as NER_FILE:
    db = json.load(NER_FILE)
    NER_FILE.close()
    
    flattend_words = []
    for edition in ['E1', 'E2', 'E3', 'E4']:
        flattend_words+=db[edition]

    flattend_words = list(map(lambda entry: {"definition": entry['definition'], "headword": entry['headword'], "type": entry['type']}, flattend_words))

    random.shuffle(flattend_words)
    others = [entry for entry in flattend_words if entry['type'] == 0]
    locations = [entry for entry in flattend_words if entry['type'] == 1]
    people = [entry for entry in flattend_words if entry['type'] == 2]

    print(len(others), len(locations), len(people))    
    
    for i in range(1,5):
        output = []
        i_start = (i-1) * 500
        i_end = i * 500
        output.append(others[i_start:i_end])
        output.append(locations[i_start:i_end])
        output.append(people[i_start:i_end])
        with open(f'./dataset/NER/batch_{i}.json', 'w', encoding='utf-8') as SAMPLE_OUT:
            json.dump(output, SAMPLE_OUT, indent=2, ensure_ascii=False)
            SAMPLE_OUT.close()
