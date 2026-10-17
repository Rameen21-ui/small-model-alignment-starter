import json, random
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42)
rows = []
for ex in ds:
    m = ex["messages"]
    if len(m) < 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant":
        continue
    ins, resp = m[0]["content"].strip(), m[1]["content"].strip()
    if not ins or not resp or len(ins) > 1500:   # long prompts would leave no room for a response
        continue
    rows.append({"instruction": ins, "response": resp})   # first turn only = single-turn
    if len(rows) == 31000:
        break
for name, part in [("train", rows[:30000]), ("val", rows[30000:])]:
    with open(f"data/sft/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
    print(name, len(part))
