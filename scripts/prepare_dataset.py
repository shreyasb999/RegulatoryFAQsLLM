# scripts/prepare_dataset.py
from datasets import load_dataset, DatasetDict, concatenate_datasets
import json

def load_seed(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    gdpr = load_seed("data/seed_qa/gdpr_seed_qas.json")
    hipaa = load_seed("data/seed_qa/hipaa_seed_qas.json")

    # Convert to HF format: {"prompt": ..., "response": ...}
    def to_records(entries):
        return [
            {
                "prompt": f"Question: {e['question']}\nAnswer:",
                "response": e["answer"]
            }
            for e in entries
        ]

    ds = concatenate_datasets([
        load_dataset("json", data_files={"train": "data/seed_qa/gdpr_seed_qas.json"}, field=None, split="train")
        .map(lambda ex: {"prompt": f"Question: {ex['question']}\nAnswer:", "response": ex["answer"]}, remove_columns=["question","answer","ref"]),
        load_dataset("json", data_files={"train": "data/seed_qa/hipaa_seed_qas.json"}, field=None, split="train")
        .map(lambda ex: {"prompt": f"Question: {ex['question']}\nAnswer:", "response": ex["answer"]}, remove_columns=["question","answer","ref"])
    ])

    # Optionally split off a small eval set
    ds = ds.train_test_split(test_size=0.1, seed=42)
    ds.save_to_disk("data/seed_dataset")
    print("Dataset prepared:", ds)

if __name__ == "__main__":
    main()
