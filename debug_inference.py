#!/usr/bin/env python3
"""Debug script to trace GLiNER weight loading."""

import torch
from pathlib import Path
from huggingface_hub import snapshot_download

print("=" * 70)
print("STEP 1: Download and inspect model files")
print("=" * 70)
model_dir = Path(snapshot_download("urchade/gliner_multi-v2.1"))
print(f"  Model dir: {model_dir}")

# Check what files exist
for f in sorted(model_dir.iterdir()):
    print(f"  {f.name} ({f.stat().st_size:,} bytes)")

# Load the saved state dict
model_file = model_dir / "model.safetensors"
if not model_file.exists():
    model_file = model_dir / "pytorch_model.bin"
print(f"\n  Loading weights from: {model_file.name}")

if model_file.suffix == ".safetensors":
    from safetensors import safe_open
    saved_state = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            saved_state[key] = f.get_tensor(key)
else:
    saved_state = torch.load(model_file, map_location="cpu")

print(f"  Saved state dict keys ({len(saved_state)}):")
for k in sorted(saved_state.keys()):
    print(f"    {k}: {saved_state[k].shape}")

print("\n" + "=" * 70)
print("STEP 2: Create model and compare keys")
print("=" * 70)
from gliner import GLiNER

# Create model without loading weights
import json
with open(model_dir / "gliner_config.json") as f:
    config_dict = json.load(f)
print(f"  Config: {json.dumps({k: v for k, v in config_dict.items() if not isinstance(v, dict)}, indent=2)}")

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

model_state = model.model.state_dict()
print(f"\n  Model state dict keys ({len(model_state)}):")
for k in sorted(model_state.keys()):
    print(f"    {k}: {model_state[k].shape}")

print("\n" + "=" * 70)
print("STEP 3: Check key mismatches")
print("=" * 70)
saved_keys = set(saved_state.keys())
model_keys = set(model_state.keys())

missing_in_saved = model_keys - saved_keys
missing_in_model = saved_keys - model_keys
common = saved_keys & model_keys

if missing_in_saved:
    print(f"\n  Keys in MODEL but NOT in saved weights ({len(missing_in_saved)}):")
    for k in sorted(missing_in_saved):
        print(f"    {k}: {model_state[k].shape}")

if missing_in_model:
    print(f"\n  Keys in SAVED but NOT in model ({len(missing_in_model)}):")
    for k in sorted(missing_in_model):
        print(f"    {k}: {saved_state[k].shape}")

if not missing_in_saved and not missing_in_model:
    print("  All keys match perfectly!")

print(f"\n  Common keys: {len(common)}")

# Check for shape mismatches
shape_mismatches = []
for k in common:
    if saved_state[k].shape != model_state[k].shape:
        shape_mismatches.append((k, saved_state[k].shape, model_state[k].shape))

if shape_mismatches:
    print(f"\n  Shape MISMATCHES ({len(shape_mismatches)}):")
    for k, s1, s2 in shape_mismatches:
        print(f"    {k}: saved={s1} vs model={s2}")
else:
    print("  No shape mismatches!")

print("\n" + "=" * 70)
print("STEP 4: Verify weights actually loaded")
print("=" * 70)
# Check span_rep_layer weights specifically
span_keys = [k for k in common if "span_rep" in k]
print(f"  Span rep layer keys: {span_keys}")
for k in span_keys:
    saved_w = saved_state[k]
    model_w = model_state[k]
    match = torch.allclose(saved_w, model_w, atol=1e-6)
    print(f"    {k}: match={match}, saved_norm={saved_w.norm():.4f}, model_norm={model_w.norm():.4f}")

# Check prompt_rep_layer
prompt_keys = [k for k in common if "prompt_rep" in k]
print(f"\n  Prompt rep layer keys: {prompt_keys}")
for k in prompt_keys:
    saved_w = saved_state[k]
    model_w = model_state[k]
    match = torch.allclose(saved_w, model_w, atol=1e-6)
    print(f"    {k}: match={match}, saved_norm={saved_w.norm():.4f}, model_norm={model_w.norm():.4f}")

# Check encoder
encoder_keys = [k for k in common if "token_rep_layer" in k]
print(f"\n  Encoder keys count: {len(encoder_keys)}")
misloaded = 0
for k in encoder_keys:
    saved_w = saved_state[k]
    model_w = model_state[k]
    if not torch.allclose(saved_w, model_w, atol=1e-6):
        misloaded += 1
        if misloaded <= 5:
            print(f"    MISMATCH: {k}")
print(f"  Encoder misloaded keys: {misloaded}/{len(encoder_keys)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
