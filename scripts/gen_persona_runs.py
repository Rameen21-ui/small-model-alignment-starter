import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
items = json.load(open("eval_sets/persona_style_eval_16.json"))
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
prompts = [tok.apply_chat_template([{"role": "user", "content": x["instruction"]}],
           tokenize=False, add_generation_prompt=True) for x in items]
def run(m, guard):
    outs = []
    for s in range(0, 16, 8):
        enc = tok(prompts[s:s+8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        kw = dict(no_repeat_ngram_size=4) if guard else {}
        with torch.no_grad():
            g = m.generate(**enc, do_sample=False, max_new_tokens=200, pad_token_id=tok.pad_token_id, **kw)
        outs += [t.strip() for t in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs
STYLE = re.compile(r"idiot|donkey|dumb|muppet|bloody|disaster|raw|stupid|silly|you gotta|!", re.I)
def stats(o):
    st = sum(bool(STYLE.search(x)) for x in o)
    g = [x.split() for x in o]
    d4 = sum(len({tuple(w[i:i+4]) for i in range(len(w)-3)}) / max(1, len(w)-3) for w in g) / len(g)
    return st, round(d4, 2)
res = {}
for guard in (True, False):
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0").eval()
    res[("base", guard)] = run(base, guard)
    for r in (1, 2, 3):
        m = PeftModel.from_pretrained(base, f"outputs/lora_persona_run{r}/adapter").eval()
        res[(f"run{r}", guard)] = run(m, guard)
        m.unload()
    m = PeftModel.from_pretrained(base, "outputs/lora_persona/adapter").eval()
    res[("run4", guard)] = run(m, guard)
print("model  guard  styled/16  distinct-4gram(1=no repetition)")
for (n, g), o in res.items(): print(n, g, *stats(o))
json.dump({f"{n}|guard={g}": o for (n, g), o in res.items()},
          open("outputs/lora_persona/all_runs_generations.json", "w"), indent=1)
for n in ("run3", "run4"):
    print("\n=====", n, "with guard =====")
    for x, o in zip(items, res[(n, True)]): print(f"[{x['category']}] {x['instruction'][:60]}\n  {o[:220]}\n")
