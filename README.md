# GLiNER NEREL: Russian NER with GLiNER

Fork of [urchade/GLiNER](https://github.com/urchade/GLiNER) fine-tuned on the [NEREL](https://github.com/nerel-ds/NEREL) dataset for Russian named entity recognition (29 entity types).

## Models

| Model | Description |
|-------|-------------|
| [fulstock/gliner-nerel-finetuned](https://huggingface.co/fulstock/gliner-nerel-finetuned) | Fine-tuned on NEREL (746 train / 94 val), 100k steps |
| [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) | Base model (upstream, multilingual) |

## Quick Start

### Install

```bash
pip install -e .
```

### Inference

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("fulstock/gliner-nerel-finetuned")
entities = model.predict_entities(
    "Иван Иванов посетил Москву 5 января 2024 года.",
    ["PERSON", "CITY", "DATE"],
    threshold=0.5,
)
for e in entities:
    print(e["text"], "=>", e["label"])
```

### Inference with sliding window (long texts)

For texts that exceed the model's token limit, `GLiNERInference` automatically splits into overlapping windows and merges results:

```python
from gliner_inference import GLiNERInference

inference = GLiNERInference(
    model_path="fulstock/gliner-nerel-finetuned",
    max_tokens=384,      # words per window
    stride_tokens=128,   # overlap between windows
)
entities = inference.predict(long_text)
# Returns: [(start_char, end_char, entity_type, entity_text), ...]
```

Batch inference collects all windows from all texts into a single batch:

```python
results = inference.predict_batch(["text1", "text2", "text3"])
```

## Entity Types

The 29 NEREL entity types (configured in `conf/nerel_labels.json`):

AGE, AWARD, CITY, COUNTRY, CRIME, DATE, DISEASE, DISTRICT, EVENT, FACILITY, FAMILY, IDEOLOGY, LANGUAGE, LAW, LOCATION, MONEY, NATIONALITY, NUMBER, ORDINAL, ORGANIZATION, PERCENT, PERSON, PENALTY, PRODUCT, PROFESSION, RELIGION, STATE\_OR\_PROVINCE, TIME, WORK\_OF\_ART

## Reproducing the Fine-tuned Model

### 1. Prepare data

Convert NEREL BRAT annotations to GLiNER JSON format:

```bash
python brat_to_gliner.py \
    --brat_path /path/to/NEREL1.1 \
    --output_path /path/to/output \
    --labels_path ./conf/nerel_labels.json
```

This produces `train.json`, `dev.json`, `test.json` with the GLiNER format:
```json
{"tokenized_text": ["word1", "word2", "..."], "ner": [[0, 2, "PERSON"], ...]}
```

### 2. Train

```bash
python train_gliner.py \
    --train_data /path/to/train.json \
    --val_data /path/to/dev.json \
    --output_dir ./saved_ckpt/nerel-finetuned \
    --base_model urchade/gliner_multi-v2.1 \
    --max_steps 100000 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --others_lr 5e-5 \
    --focal_loss_alpha 0.75 \
    --focal_loss_gamma 0.0
```

See `conf/train_config.json` for the full set of hyperparameters used.

### 3. Evaluate

```bash
python gliner_inference.py \
    --input /path/to/test.json \
    --model ./saved_ckpt/nerel-finetuned \
    --metrics_output metrics.json \
    --measure_time --timing_output timing.json
```

Compare multiple models on BRAT-format test data:

```bash
python gliner_inference.py --compare \
    --brat_folder /path/to/test \
    --models urchade/gliner_multi-v2.1 ./saved_ckpt/nerel-finetuned \
    --names "Pretrained" "Finetuned"
```

## Repository Structure

| File | Purpose |
|------|---------|
| `gliner_inference.py` | Inference wrapper with sliding window, batch support, evaluation CLI |
| `train_gliner.py` | Fine-tuning script |
| `brat_to_gliner.py` | BRAT annotation format to GLiNER JSON converter |
| `metrics_to_csv.py` | Convert evaluation metrics JSON to CSV |
| `conf/` | Label configs and training/inference config templates |
| `gliner/` | GLiNER library (forked from upstream) |

## Credits

GLiNER was originally developed by Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois. See the [original repository](https://github.com/urchade/GLiNER) and [paper](https://arxiv.org/abs/2311.08526).
