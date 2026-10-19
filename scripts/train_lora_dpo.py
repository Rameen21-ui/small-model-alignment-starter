import json, os, random, inspect, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT = "outputs/lora_persona"
rows = json.load(open("data/persona_lora/ramsay_style_preferences.json"))
random.Random(42).shuffle(rows)
conv = lambda r: {"prompt": [{"role": "user", "content": r["prompt"]}],
                  "chosen": [{"role": "assistant", "content": r["chosen"]}],
                  "rejected": [{"role": "assistant", "content": r["rejected"]}]}
val_ds = Dataset.from_list([conv(r) for r in rows[:20]])
train_ds = Dataset.from_list([conv(r) for r in rows[20:]])
print("train", len(train_ds), "val", len(val_ds))

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])

want = dict(output_dir=OUT, per_device_train_batch_size=2, gradient_accumulation_steps=4,
            per_device_eval_batch_size=4, num_train_epochs=3, learning_rate=7e-5,
            lr_scheduler_type="cosine", warmup_ratio=0.1, beta=0.3, fp16=True,
            max_length=512, max_prompt_length=256, logging_steps=2,
            eval_strategy="steps", evaluation_strategy="steps", eval_steps=12,
            save_strategy="no", report_to="none", seed=42, gradient_checkpointing=False)
ok = set(inspect.signature(DPOConfig.__init__).parameters)
print("dropped (not in this TRL):", [k for k in want if k not in ok])
cfg = DPOConfig(**{k: v for k, v in want.items() if k in ok})

kw = dict(model=model, ref_model=None, args=cfg, train_dataset=train_ds,
          eval_dataset=val_ds, peft_config=lora)
try:
    trainer = DPOTrainer(processing_class=tok, **kw)
except TypeError:
    trainer = DPOTrainer(tokenizer=tok, **kw)

trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(f"{OUT}/adapter")
json.dump(trainer.state.log_history, open(f"{OUT}/metrics.json", "w"), indent=1)
print("saved adapter")
