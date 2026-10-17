import torch
import torch.nn.functional as F


def _encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_sft_batch(tokenizer, prompts, responses, max_length):
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = eos if eos is not None else 0

    rows_ids, rows_labels = [], []
    for p, r in zip(prompts, responses):
        p_ids = _encode(tokenizer, p)
        r_ids = _encode(tokenizer, r)
        if eos is not None:
            r_ids = r_ids + [eos]                      # EOS BEFORE truncation
        rows_ids.append((p_ids + r_ids)[:max_length])
        rows_labels.append(([-100] * len(p_ids) + r_ids)[:max_length])

    B, L = len(rows_ids), max(len(x) for x in rows_ids)
    input_ids = torch.full((B, L), pad, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(rows_ids, rows_labels)):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(lab, dtype=torch.long)
        attention_mask[i, :n] = 1                      # from LENGTH, not token id
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _token_logps(model, batch):
    """Returns per-token logps [B, L-1] and mask [B, L-1] (True = response token)."""
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"]).logits   # [B, L, V]
    logits = logits[:, :-1].float()                                 # [B, L-1, V]
    targets = batch["labels"][:, 1:]                                # [B, L-1]
    mask = targets.ne(-100)
    safe = targets.masked_fill(~mask, 0)                            # avoid index -100
    tok_logit = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)   # [B, L-1]
    token_logp = tok_logit - torch.logsumexp(logits, dim=-1)        # [B, L-1]
    return token_logp, mask


def _response_logps(model, batch):
    token_logp, mask = _token_logps(model, batch)
    return (token_logp * mask).sum(dim=-1)                          # [B]


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return _response_logps(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    # Reduction: mean over ALL unmasked response tokens in the batch.
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    token_logp, mask = _token_logps(model, batch)
    n = mask.sum().clamp(min=1)                                     # guard: 0 response tokens
    return -(token_logp * mask).sum() / n                           # scalar


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length),
    }


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    pi_gap = policy_chosen_logps - policy_rejected_logps            # [B]
    ref_gap = (reference_chosen_logps - reference_rejected_logps).detach()  # [B]
    return -F.logsigmoid(beta * (pi_gap - ref_gap)).mean()          # scalar


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    pol_c = _response_logps(policy_model, batch["chosen"])
    pol_r = _response_logps(policy_model, batch["rejected"])
    with torch.no_grad():
        ref_c = _response_logps(reference_model, batch["chosen"]).to(pol_c.device)
        ref_r = _response_logps(reference_model, batch["rejected"]).to(pol_c.device)
    return compute_dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, beta)
