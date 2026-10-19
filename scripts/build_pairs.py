import json, random, os
rng = random.Random(42)
os.makedirs("outputs/eval", exist_ok=True)

def build(items_a, items_b, name_a, name_b, out):
    b_by_id = {r["id"]: r for r in items_b}
    rows = []
    for r in items_a:
        o = b_by_id[r["id"]]
        flip = rng.random() < 0.5                      # seeded coin flip per pair
        first, second = (o, r) if flip else (r, o)
        names = (name_b, name_a) if flip else (name_a, name_b)
        rows.append({"id": r["id"], "instruction": r["instruction"],
                     "response_a": first["output"], "response_b": second["output"],
                     "_meta": {"A": names[0], "B": names[1]}})   # hidden from the judge
    with open(out, "w") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print(out, len(rows), "model1 in slot A:", sum(x["_meta"]["A"] == name_a for x in rows))

L = lambda p: json.load(open(p))
build(L("outputs/dpo_hh/hh_dpo_hh.json"), L("outputs/sft/hh_sft.json"), "dpo", "sft", "outputs/eval/hh_dpo_vs_sft_pairs.jsonl")
build(L("outputs/dpo_uf/alpaca_dpo_uf.json"), L("outputs/sft/alpaca_sft.json"), "dpo", "sft", "outputs/eval/uf_dpo_vs_sft_pairs.jsonl")

p = L("outputs/lora_persona/persona_generations.json")
a = [{"id": x["id"], "instruction": x["instruction"], "output": x["lora"]} for x in p]
b = [{"id": x["id"], "instruction": x["instruction"], "output": x["base"]} for x in p]
build(a, b, "lora", "base", "outputs/eval/persona_lora_vs_base_pairs.jsonl")
