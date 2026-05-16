# Skill NER — Job Skill Extraction with DistilBERT

A fine-tuned NER model that extracts IT skills from job descriptions.
Trained on real recruitment data using HuggingFace Transformers.

## What it does
- Detects skill mentions in raw job description text (e.g. `python`, `machine learning`, `gcp`)
- Normalizes synonyms — `ReactJS`, `react.js`, `react js` all map to `react`
- Returns a clean deduplicated list of skills per job posting

## How it works
1. **Synonym map** — fuzzy clustering + manual rules canonicalize skill variants
2. **Span detection** — finds skill mentions using both raw and canonical forms
3. **BIO tagging** — labels tokens as `B-SKILL`, `I-SKILL`, or `O`
4. **DistilBERT fine-tuning** — `distilbert-base-uncased` + token classification head
5. **Undersampling** — balances O-heavy sequences to reduce majority class bias

## Results
| Metric | Score |
|-----------|-------|
| Precision | 0.86 |
| Recall | 0.90 |
| F1 | **0.88** |

## Quick start
Rerun the code first and do the following:

```python
from transformers import pipeline

ner = pipeline("token-classification",
               model="./skill-ner/checkpoint-445",
               aggregation_strategy="max")

def extract_skills(text, threshold=0.3):
    return list({e["word"].strip().lower()
                 for e in ner(text)
                 if e["entity_group"] == "SKILL" and e["score"] >= threshold})

print(extract_skills("Looking for a Python developer with AWS and React experience."))
# ['python', 'aws', 'react']
```

## Stack
`transformers` · `datasets` · `seqeval` · `rapidfuzz` · `spacy` · `torch`
