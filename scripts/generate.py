import argparse, json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(__file__))
from common import format_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--eval_path", required=True)
ap.add_argument("--out_path", required=True)
ap.add_argument("--generator", required=True)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--max_new_tokens", type=int, default=256)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda:0").eval()

items = json.load(open(a.eval_path))
if a.limit: items = items[:a.limit]
prompts = [format_prompt(x["instruction"]) for x in items]
order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
outputs = [None] * len(items)

for s in range(0, len(order), a.batch_size):
    ids = order[s:s + a.batch_size]
    enc = tok([prompts[i] for i in ids], return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    for i, g in zip(ids, tok.batch_decode(gen, skip_special_tokens=True)):
        outputs[i] = g.split("\n\n###")[0].strip()
    print(f"{min(s + a.batch_size, len(order))}/{len(order)}", flush=True)

res = [{"id": x["id"], "instruction": x["instruction"], "output": o,
        "generator": a.generator, "dataset": x["dataset"]} for x, o in zip(items, outputs)]
os.makedirs(os.path.dirname(a.out_path), exist_ok=True)
json.dump(res, open(a.out_path, "w"), indent=1)
print("saved", a.out_path)
