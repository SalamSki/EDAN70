import json
import random

with open('ner_entries.json') as f:
    d = json.load(f)
    d['words'] = []
    for entry in d['E1']:
        d['words'].append(entry)
    for entry in d['E2']:
        d['words'].append(entry)
    for entry in d['E3']:
        d['words'].append(entry)
    for entry in d['E4']:
        d['words'].append(entry)

    d['words'] = list(map(lambda entry: {"definition": entry['definition'], "type": entry['type']}, d['words']))

    entries = sorted(d['words'], key=lambda x: random.random())
    others = [entry for entry in entries if entry['type'] == 0]
    locations = [entry for entry in entries if entry['type'] == 1]
    people = [entry for entry in entries if entry['type'] == 2]
    
    output = {}
    output['data'] = []
    output['data'].append(others[:333])
    output['data'].append(locations[:333])
    output['data'].append(people[:333])
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
