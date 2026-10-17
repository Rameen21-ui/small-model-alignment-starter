import json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
MAX_PROMPT, MAX_TOTAL = 256, 512
os.makedirs("data/dpo", exist_ok=True)

def ntok(t): return len(tok.encode(t, add_special_tokens=False))

def ok(ins, ch, rj):
    if not ins or not ch or not rj or ch == rj: return False
    if len(ins) > 1500: return False
    p = ntok("### Instruction:\n" + ins + "\n\n### Response:\n")
    return p <= MAX_PROMPT and p + min(ntok(ch), ntok(rj)) < MAX_TOTAL

def hh_parse(t):
    # keep only single-turn: exactly one Human and one Assistant turn
    if t.count("\n\nHuman:") != 1 or t.count("\n\nAssistant:") != 1: return None
    h, a = t.split("\n\nAssistant:")
    return h.replace("\n\nHuman:", "").strip(), a.strip()

def hh_rows(split):
    for ex in load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=42):
        c, r = hh_parse(ex["chosen"]), hh_parse(ex["rejected"])
        if c is None or r is None or c[0] != r[0]: continue
        yield c[0], c[1], r[1]

def uf_rows(split):
    for ex in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).shuffle(seed=42):
        c, r = ex["chosen"], ex["rejected"]
        if len(c) != 2 or len(r) != 2: continue
        yield c[0]["content"].strip(), c[1]["content"].strip(), r[1]["content"].strip()

def build(name, gen_fn, train_split, val_split, n_train=15000, n_val=1000):
    for part, split, n in [("val", val_split, n_val), ("train", train_split, n_train)]:
        rows = []
        for ins, ch, rj in gen_fn(split):
            if ok(ins, ch, rj):
                rows.append({"instruction": ins, "chosen": ch, "rejected": rj})
                if len(rows) == n: break
        with open(f"data/dpo/{name}_{part}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")

build("hh", hh_rows, "train", "test")
build("uf", uf_rows, "train_prefs", "test_prefs")
