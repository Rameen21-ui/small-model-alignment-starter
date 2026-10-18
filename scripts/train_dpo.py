import argparse, json, os, random, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "scripts")
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import format_prompt
import submission_adapter as A            # same graded code: logps + DPO loss

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)             # hh or uf
ap.add_argument("--out_dir", required=True)
ap.add_argument("--init", default="outputs/sft/best")
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--max_steps", type=int, default=None)
ap.add_argument("--eval_every", type=int, default=117)
ap.add_argument("--val_n", type=int, default=1000)
ap.add_argument("--grad_ckpt", action="store_true")
a = ap.parse_args()

SEED, ACCUM, MAXLEN, CLIP = 42, 32, 512, 1.0
random.seed(SEED); torch.manual_seed(SEED)
load = lambda p: [json.loads(l) for l in open(p)]
train, val = load(f"data/dpo/{a.data}_train.jsonl"), load(f"data/dpo/{a.data}_val.jsonl")
random.Random(SEED).shuffle(train)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
policy = AutoModelForCausalLM.from_pretrained(a.init, torch_dtype=torch.float32).to("cuda:0")
ref = AutoModelForCausalLM.from_pretrained(a.init, torch_dtype=torch.float32).to("cuda:1").eval()
for p in ref.parameters(): p.requires_grad_(False)
policy.config.use_cache = False
if a.grad_ckpt:
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
policy.train()

opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0)  
scaler = torch.amp.GradScaler("cuda")
total = len(train) // ACCUM                                         
if a.max_steps: total = min(total, a.max_steps)

def fmt(rows):
    return ([format_prompt(r["instruction"]) for r in rows],
            [r["chosen"] for r in rows], [r["rejected"] for r in rows])

def pair_logps(p, c, j):
    with torch.autocast("cuda", dtype=torch.float16):
        pc = A.compute_response_logprobs(policy, tok, p, c, MAXLEN)     
        pr = A.compute_response_logprobs(policy, tok, p, j, MAXLEN)
        with torch.no_grad():                                           
            rc = A.compute_response_logprobs(ref, tok, p, c, MAXLEN).to("cuda:0")
            rr = A.compute_response_logprobs(ref, tok, p, j, MAXLEN).to("cuda:0")
    return pc, pr, rc, rr

@torch.no_grad()
def validate():
    policy.eval(); L, W, M = [], [], []
    vs = val[:a.val_n]
    for s in range(0, len(vs), 4):
        pc, pr, rc, rr = pair_logps(*fmt(vs[s:s + 4]))
        rw, rl = a.beta * (pc - rc), a.beta * (pr - rr)
        L.append(-F.logsigmoid(rw - rl)); W.append((rw > rl).float()); M.append(rw - rl)
    policy.train()
    return (torch.cat(L).mean().item(), torch.cat(W).mean().item(), torch.cat(M).mean().item())

os.makedirs(a.out_dir, exist_ok=True)
log = open(f"{a.out_dir}/metrics.jsonl", "a")
def write(rec):
    log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)

t0 = time.time()
vl, va, vm = validate()                                
write({"step": 0, "val_loss": vl, "val_acc": va, "val_margin": vm, "min": 0})
best = (-1.0, 0.0)
for step in range(total):
    sl = sa = sm = 0.0
    for m in range(ACCUM):
        pc, pr, rc, rr = pair_logps(*fmt([train[step * ACCUM + m]]))     
        loss = A.compute_dpo_loss_from_logps(pc, pr, rc, rr, a.beta) / ACCUM
        scaler.scale(loss).backward()
        with torch.no_grad():
            rw, rl = a.beta * (pc - rc), a.beta * (pr - rr)
            sl += loss.item(); sa += (rw > rl).float().item() / ACCUM; sm += (rw - rl).item() / ACCUM
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP).item()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    rec = {"step": step + 1, "train_loss": sl, "train_acc": sa, "train_margin": sm, "grad_norm": gn,
           "min": round((time.time() - t0) / 60, 1)}
    if (step + 1) % a.eval_every == 0 or step + 1 == total:
        vl, va, vm = validate()
        rec.update(val_loss=vl, val_acc=va, val_margin=vm)
        if (va, -vl) > (best[0], -best[1]):
            best = (va, vl)
            policy.save_pretrained(f"{a.out_dir}/best"); tok.save_pretrained(f"{a.out_dir}/best")
    if (step + 1) % 5 == 0 or "val_loss" in rec: write(rec)
    else: log.write(json.dumps(rec) + "\n"); log.flush()
policy.save_pretrained(f"{a.out_dir}/final"); tok.save_pretrained(f"{a.out_dir}/final")
print("done, best val acc/loss:", best)
