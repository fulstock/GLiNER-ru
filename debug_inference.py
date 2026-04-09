#!/usr/bin/env python3
"""Debug script to trace GLiNER inference step by step."""

import torch
from gliner import GLiNER

text = "Владимир Путин посетил Москву."
labels = ["PERSON", "CITY"]
threshold = 0.01

print("=" * 70)
print("STEP 1: Load model")
print("=" * 70)
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
model.eval()

print(f"  Model class: {model.__class__.__name__}")
print(f"  Inner model class: {model.model.__class__.__name__}")
print(f"  Device: {model.device}")
print(f"  Config model_name: {model.config.model_name}")
print(f"  Config span_mode: {model.config.span_mode}")
print(f"  Config max_width: {model.config.max_width}")
print(f"  Config hidden_size: {model.config.hidden_size}")

print("\n" + "=" * 70)
print("STEP 2: Prepare inputs (tokenize)")
print("=" * 70)
tokens, starts, ends = model.prepare_inputs([text])
print(f"  Tokens: {tokens}")
print(f"  Start indices: {starts}")
print(f"  End indices: {ends}")

input_x = model.prepare_base_input(tokens)
print(f"  input_x: {input_x}")

print("\n" + "=" * 70)
print("STEP 3: Collate batch")
print("=" * 70)
collator = model.data_collator_class(
    model.config,
    data_processor=model.data_processor,
    return_tokens=True,
    return_entities=True,
    return_id_to_classes=True,
    prepare_labels=False,
)
entity_types = list(dict.fromkeys(labels))
batch = collator(input_x, entity_types=entity_types)

print(f"  Batch keys: {list(batch.keys())}")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.numel() < 50:
            print(f"    values: {v}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n" + "=" * 70)
print("STEP 4: Encoder forward (token_rep_layer)")
print("=" * 70)
device = model.device
batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

input_ids = batch_gpu["input_ids"]
attention_mask = batch_gpu["attention_mask"]
text_lengths = batch_gpu.get("text_lengths")
words_mask = batch_gpu.get("words_mask")

print(f"  input_ids shape: {input_ids.shape}")
print(f"  input_ids[0, :30]: {input_ids[0, :30].tolist()}")
print(f"  attention_mask shape: {attention_mask.shape}")
print(f"  attention_mask[0, :30]: {attention_mask[0, :30].tolist()}")

# Decode input_ids to see what the tokenizer produced
tokenizer = model.data_processor.transformer_tokenizer
decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
# Filter out padding
non_pad = [t for t in decoded_tokens if t != tokenizer.pad_token]
print(f"  Decoded tokens (non-pad): {non_pad}")

with torch.no_grad():
    token_embeds = model.model.token_rep_layer(input_ids, attention_mask)

print(f"  token_embeds shape: {token_embeds.shape}")
print(f"  token_embeds stats: min={token_embeds.min():.4f}, max={token_embeds.max():.4f}, mean={token_embeds.mean():.4f}, std={token_embeds.std():.4f}")
print(f"  token_embeds[0, 0, :10]: {token_embeds[0, 0, :10].tolist()}")

print("\n" + "=" * 70)
print("STEP 5: Extract prompts + word embeddings")
print("=" * 70)
with torch.no_grad():
    prompts_emb, prompts_mask, words_emb, mask = model.model.get_representations(
        input_ids, attention_mask, text_lengths, words_mask
    )

print(f"  prompts_emb shape: {prompts_emb.shape}")
print(f"  prompts_emb stats: min={prompts_emb.min():.4f}, max={prompts_emb.max():.4f}, mean={prompts_emb.mean():.4f}")
print(f"  prompts_mask: {prompts_mask}")
print(f"  words_emb shape: {words_emb.shape}")
print(f"  words_emb stats: min={words_emb.min():.4f}, max={words_emb.max():.4f}, mean={words_emb.mean():.4f}")
print(f"  mask: {mask}")

print("\n" + "=" * 70)
print("STEP 6: Span representations")
print("=" * 70)
span_idx = batch_gpu["span_idx"]
span_mask = batch_gpu["span_mask"]
print(f"  span_idx shape: {span_idx.shape}")
print(f"  span_mask shape: {span_mask.shape}")
print(f"  span_mask sum (valid spans): {span_mask.sum().item()}")

target_W = span_idx.size(1) // model.config.max_width
words_emb_fit, mask_fit = model.model._fit_length(words_emb, mask, target_W)
print(f"  words_emb after fit: {words_emb_fit.shape}")

span_idx_masked = span_idx * span_mask.unsqueeze(-1)
with torch.no_grad():
    span_rep = model.model.span_rep_layer(words_emb_fit, span_idx_masked)
print(f"  span_rep shape: {span_rep.shape}")
print(f"  span_rep stats: min={span_rep.min():.4f}, max={span_rep.max():.4f}, mean={span_rep.mean():.4f}")

print("\n" + "=" * 70)
print("STEP 7: Compute scores (einsum)")
print("=" * 70)
with torch.no_grad():
    prompts_proj = model.model.prompt_rep_layer(prompts_emb)
print(f"  prompts_proj shape: {prompts_proj.shape}")
print(f"  prompts_proj stats: min={prompts_proj.min():.4f}, max={prompts_proj.max():.4f}, mean={prompts_proj.mean():.4f}")

scores = torch.einsum("BLKD,BCD->BLKC", span_rep, prompts_proj)
print(f"  scores shape: {scores.shape}")
print(f"  scores stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, std={scores.std():.4f}")

# Apply sigmoid to see probabilities
probs = torch.sigmoid(scores)
print(f"  probs (sigmoid) stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
print(f"  probs > 0.3: {(probs > 0.3).sum().item()}")
print(f"  probs > 0.1: {(probs > 0.1).sum().item()}")
print(f"  probs > 0.01: {(probs > 0.01).sum().item()}")
print(f"  Top 10 probs: {probs.flatten().topk(10).values.tolist()}")

print("\n" + "=" * 70)
print("STEP 8: Full model forward (sanity check)")
print("=" * 70)
with torch.no_grad():
    model_output = model.model(**batch_gpu, threshold=threshold)

logits = model_output.logits
print(f"  logits shape: {logits.shape}")
print(f"  logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
logits_probs = torch.sigmoid(logits)
print(f"  logits probs max: {logits_probs.max():.4f}")
print(f"  Top 10 logits probs: {logits_probs.flatten().topk(10).values.tolist()}")

print("\n" + "=" * 70)
print("STEP 9: Decode")
print("=" * 70)
decoded = model.decoder.decode(
    batch["tokens"],
    batch["id_to_classes"],
    logits,
    span_idx=model_output.span_idx,
    span_mask=model_output.span_mask,
    span_logits=model_output.span_logits,
    flat_ner=True,
    threshold=threshold,
    multi_label=False,
)
print(f"  Decoded output: {decoded}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
