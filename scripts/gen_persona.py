import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
items = json.load(open("eval_sets/persona_style_eval_16.json"))
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
prompts = [tok.apply_chat_template([{"role": "user", "content": x["instruction"]}],
                                   tokenize=False, add_generation_prompt=True) for x in items]

def run(model):
    outs = []
    for s in range(0, len(prompts), 8):
        enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=False, max_new_tokens=200, no_repeat_ngram_size=4, pad_token_id=tok.pad_token_id)
        outs += [t.strip() for t in tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return outs

base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0").eval()
base_out = run(base)
lora = PeftModel.from_pretrained(base, "outputs/lora_persona/adapter").eval()
lora_out = run(lora)
res = [dict(x, base=b, lora=l) for x, b, l in zip(items, base_out, lora_out)]
json.dump(res, open("outputs/lora_persona/persona_generations.json", "w"), indent=1)
for r in res[::4]:
    print("[%s] %s\n BASE: %s\n LORA: %s\n" % (r["category"], r["instruction"], r["base"][:250], r["lora"][:250]))
