import argparse, json, math, os, random, sys, time, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import format_prompt
import submission_adapter as A          # the SAME loss code the grader tests

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default="outputs/sft")
ap.add_argument("--max_steps", type=int, default=None)   # for smoke tests
ap.add_argument("--eval_every", type=int, default=125)
a = ap.parse_args()

SEED, MB, ACCUM, LR, WD, CLIP, WARM, MAXLEN = 42, 2, 16, 2e-5, 0.1, 1.0, 0.03, 512
random.seed(SEED); torch.manual_seed(SEED)

load = lambda p: [json.loads(l) for l in open(p)]
train, val = load("data/sft/train.jsonl"), load("data/sft/val.jsonl")
random.Random(SEED).shuffle(train)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).to("cuda:0")
model.config.use_cache = False
model.train()

decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD},
                         {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(0.9, 0.95))
total = len(train) // (MB * ACCUM)                      # 30000 // 32 = 937
if a.max_steps: total = min(total, a.max_steps)
warm = max(1, int(WARM * total))
def lr_at(s):
    if s < warm: return (s + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
scaler = torch.amp.GradScaler("cuda")

def fmt(rows): return [format_prompt(r["instruction"]) for r in rows], [r["response"] for r in rows]

@torch.no_grad()
def validate():
    model.eval(); tot, n = 0.0, 0
    for s in range(0, len(val), 4):
        p, r = fmt(val[s:s + 4])
        with torch.autocast("cuda", dtype=torch.float16):
            tot += A.compute_sft_loss(model, tok, p, r, MAXLEN).item(); n += 1
    model.train(); return tot / n

os.makedirs(a.out_dir, exist_ok=True)
best, t0 = float("inf"), time.time()
log = open(f"{a.out_dir}/metrics.jsonl", "a")
for step in range(total):
    run = 0.0
    for m in range(ACCUM):
        s = (step * ACCUM + m) * MB
        p, r = fmt(train[s:s + MB])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = A.compute_sft_loss(model, tok, p, r, MAXLEN) / ACCUM   # divide by ACCUM
        scaler.scale(loss).backward()
        run += loss.item()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP).item()
    scaler.step(opt); scaler.update(); sched.step()
    opt.zero_grad(set_to_none=True)                     # once per optimizer step
    rec = {"step": step + 1, "train_loss": run, "lr": sched.get_last_lr()[0], "grad_norm": gn}
    if (step + 1) % a.eval_every == 0 or step + 1 == total:
        v = validate(); rec["val_loss"] = v
        if v < best:
            best = v; model.save_pretrained(f"{a.out_dir}/best"); tok.save_pretrained(f"{a.out_dir}/best")
    log.write(json.dumps(rec) + "\n"); log.flush()
    if (step + 1) % 10 == 0 or "val_loss" in rec:
        print(rec, f"{(time.time()-t0)/60:.1f}min", flush=True)
model.save_pretrained(f"{a.out_dir}/final"); tok.save_pretrained(f"{a.out_dir}/final")
print("done, best val", best)
