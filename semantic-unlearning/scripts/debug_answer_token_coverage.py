#!/usr/bin/env python3
import json
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

CONFIG = "/scratch/yl258/kp759/Unlearning/semantic-unlearning/config/config_3b_instruct_forget05.yaml"
TOKEN_JSON = "outputs/semantic_tokens.json"

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

with open(TOKEN_JSON) as f:
    tok_data = json.load(f)

selected = set(map(int, tok_data["token_ids"]))

tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
forget_ds = load_dataset("locuslab/TOFU", name="forget05", split="train")

total = 0
covered = 0
missing_counter = {}

for row in forget_ds:
    answer = " " + row["answer"].strip()
    ids = tokenizer.encode(answer, add_special_tokens=False)
    for tid in ids:
        total += 1
        if tid in selected:
            covered += 1
        else:
            missing_counter[tid] = missing_counter.get(tid, 0) + 1

print("Answer-token coverage:", covered / max(total, 1))
print("covered:", covered, "total:", total)

print("\nTop missing answer tokens:")
for tid, c in sorted(missing_counter.items(), key=lambda x: -x[1])[:80]:
    print(tid, repr(tokenizer.decode([tid])), c)
    