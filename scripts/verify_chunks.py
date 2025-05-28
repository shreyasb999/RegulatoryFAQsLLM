import json
import random
from statistics import median

def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def word_count(text):
    return len(text.split())

def inspect(path, sample_size=5):
    chunks = load_chunks(path)
    counts = [word_count(c['text']) for c in chunks]
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Word-count stats → min: {min(counts)}, median: {median(counts)}, max: {max(counts)}\n")
    
    # sort by word count only
    extremes = sorted(zip(counts, chunks), key=lambda x: x[0])
    
    print("=== Shortest chunk ===")
    print(extremes[0][1]['text'], "\n")
    print("=== Longest chunk ===")
    print(extremes[-1][1]['text'], "\n")
    
    print(f"=== {sample_size} Random Samples ===")
    for wc, chunk in random.sample(extremes, sample_size):
        snippet = chunk['text'].replace('\n', ' ')
        print(f"(~{wc} words) {snippet[:300]}…\n")

if __name__ == '__main__':
    print("GDPR chunks:")
    inspect("data/parsed/gdpr.json")
    print("\nHIPAA chunks:")
    inspect("data/parsed/hipaa.json")
