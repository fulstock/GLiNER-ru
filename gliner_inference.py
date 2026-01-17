#!/usr/bin/env python3
"""
GLiNER Inference Module.

Provides GLiNERInference class with the same interface as BinderInference
for named entity recognition on Russian text.

Usage (as library):
    from gliner_inference import GLiNERInference

    inference = GLiNERInference(model_path="urchade/gliner_multi-v2.1")
    entities = inference.predict("Владимир Путин посетил Москву 5 января 2024 года.")
    # Returns: [(0, 14, 'PERSON', 'Владимир Путин'), (24, 30, 'CITY', 'Москву'), ...]

Usage (CLI evaluation):
    python gliner_inference.py --input test.json --output predictions.json --metrics_output metrics.json
"""

import argparse
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
from gliner import GLiNER
from tqdm import tqdm


# Default NEREL entity labels
DEFAULT_LABELS = [
    "AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISEASE", "DISTRICT",
    "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION",
    "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PERCENT",
    "PERSON", "PENALTY", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE",
    "TIME", "WORK_OF_ART"
]


class GLiNERInference:
    """
    GLiNER inference class with BinderInference-compatible interface.

    Provides predict() and predict_batch() methods that return entities
    in the same format as BinderInference:
        (start_char, end_char, entity_type, entity_text)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
        labels: Optional[List[str]] = None,
        labels_path: Optional[str] = None,
        flat_ner: bool = True
    ):
        """
        Initialize GLiNER inference.

        Args:
            model_path: Path to GLiNER model or HuggingFace model ID.
                       Default: "urchade/gliner_multi-v2.1"
            device: Device to use ("auto", "cpu", "cuda", "cuda:0", etc.)
            threshold: Prediction confidence threshold (0.0-1.0)
            labels: List of entity labels to predict.
                   Default: NEREL 29 entity types
            labels_path: Path to JSON file with labels config
            flat_ner: If True, no overlapping entities. If False, allow nested.
        """
        self.model_path = model_path or "urchade/gliner_multi-v2.1"
        self.threshold = threshold
        self.flat_ner = flat_ner

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load labels
        self.labels = self._load_labels(labels, labels_path)

        # Lazy model loading
        self._model: Optional[GLiNER] = None

    def _load_labels(
        self,
        labels: Optional[List[str]],
        labels_path: Optional[str]
    ) -> List[str]:
        """Load entity labels from config or use defaults."""
        if labels is not None:
            return labels

        if labels_path is not None and os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='UTF-8') as f:
                config = json.load(f)
                return config.get("labels", DEFAULT_LABELS)

        # Try default config path
        default_path = os.path.join(
            os.path.dirname(__file__), "conf", "nerel_labels.json"
        )
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='UTF-8') as f:
                config = json.load(f)
                return config.get("labels", DEFAULT_LABELS)

        return DEFAULT_LABELS

    @property
    def model(self) -> GLiNER:
        """Lazy-load the GLiNER model."""
        if self._model is None:
            print(f"Loading GLiNER model from {self.model_path}...")
            self._model = GLiNER.from_pretrained(self.model_path)
            self._model = self._model.to(self.device)
            self._model.eval()
            print(f"Model loaded on {self.device}")
        return self._model

    def predict(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Perform named entity recognition on input text.

        Args:
            text: Input text string

        Returns:
            List of (start_char, end_char, entity_type, entity_text) tuples,
            sorted by position.
        """
        if not text or not text.strip():
            return []

        # Predict entities using GLiNER
        entities = self.model.predict_entities(
            text,
            self.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner
        )

        # Convert to BinderInference format
        results = []
        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']
            entity_type = entity['label']
            entity_text = entity['text']
            results.append((start_char, end_char, entity_type, entity_text))

        # Sort by position
        results.sort(key=lambda x: (x[0], x[1]))

        return results

    def predict_batch(
        self,
        texts: Union[List[str], Tuple[str, ...], str]
    ) -> List[List[Tuple[int, int, str, str]]]:
        """
        Perform batch prediction on multiple texts.

        Args:
            texts: List/tuple of text strings, or single text string

        Returns:
            List of entity lists, one per input text.
            Each entity is (start_char, end_char, entity_type, entity_text).
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        if len(texts) == 1:
            return [self.predict(texts[0])]

        # Batch prediction using GLiNER's inference method
        all_entities = self.model.inference(
            texts,
            self.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner
        )

        # Convert to BinderInference format
        results = []
        for text_entities in all_entities:
            text_results = []
            for entity in text_entities:
                start_char = entity['start']
                end_char = entity['end']
                entity_type = entity['label']
                entity_text = entity['text']
                text_results.append((start_char, end_char, entity_type, entity_text))

            # Sort by position
            text_results.sort(key=lambda x: (x[0], x[1]))
            results.append(text_results)

        return results

    def warm_up(self, sample_text: Optional[str] = None) -> None:
        """
        Warm up the model by running a sample prediction.

        Args:
            sample_text: Text to use for warm-up.
                        Default: Russian sample text
        """
        if sample_text is None:
            sample_text = "Владимир Путин встретился с президентом США в Москве."

        _ = self.predict(sample_text)
        print("Model warmed up successfully")

    def get_entity_types(self) -> List[str]:
        """Return list of supported entity types."""
        return self.labels.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "threshold": self.threshold,
            "flat_ner": self.flat_ner,
            "entity_types": self.labels,
            "num_entity_types": len(self.labels),
            "model_loaded": self._model is not None
        }

    def set_threshold(self, threshold: float) -> None:
        """Update prediction threshold."""
        self.threshold = threshold

    def set_labels(self, labels: List[str]) -> None:
        """Update entity labels to predict."""
        self.labels = labels


# =============================================================================
# Timing utilities
# =============================================================================

class TimingStats:
    """Collect timing statistics."""

    def __init__(self):
        self.times: Dict[str, List[float]] = defaultdict(list)
        self.start_time: Optional[float] = None

    def start(self):
        """Start total timing."""
        self.start_time = time.perf_counter()

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure time for a section."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)

    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        total_time = time.perf_counter() - self.start_time if self.start_time else 0

        stats = {
            "total_sec": round(total_time, 3)
        }

        for name, times in self.times.items():
            if len(times) == 1:
                stats[f"{name}_sec"] = round(times[0], 3)
            else:
                stats[f"{name}_total_sec"] = round(sum(times), 3)
                stats[f"{name}_avg_sec"] = round(sum(times) / len(times), 4)
                stats[f"{name}_min_sec"] = round(min(times), 4)
                stats[f"{name}_max_sec"] = round(max(times), 4)
                stats[f"{name}_count"] = len(times)

        return stats


# =============================================================================
# Data conversion utilities
# =============================================================================

def tokens_to_text_with_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Reconstruct text from tokens and compute character offsets.

    Returns:
        text: Reconstructed text (tokens joined with spaces)
        offsets: List of (start_char, end_char) for each token
    """
    offsets = []
    current_pos = 0
    text_parts = []

    for i, token in enumerate(tokens):
        start = current_pos
        end = current_pos + len(token)
        offsets.append((start, end))
        text_parts.append(token)
        current_pos = end + 1  # +1 for space

    text = " ".join(text_parts)
    return text, offsets


def token_indices_to_char_span(
    start_idx: int,
    end_idx: int,
    offsets: List[Tuple[int, int]],
    text: str
) -> Tuple[int, int, str]:
    """
    Convert token indices (inclusive) to character span.

    Returns:
        (start_char, end_char, entity_text)
    """
    if start_idx < 0 or end_idx >= len(offsets):
        return (-1, -1, "")

    start_char = offsets[start_idx][0]
    end_char = offsets[end_idx][1]
    entity_text = text[start_char:end_char]

    return start_char, end_char, entity_text


def convert_gold_entities(
    ner_annotations: List[List],
    offsets: List[Tuple[int, int]],
    text: str
) -> List[Tuple[int, int, str, str]]:
    """
    Convert gold NER annotations from token indices to character spans.

    Args:
        ner_annotations: List of [start_idx, end_idx, entity_type]
        offsets: Token character offsets
        text: Reconstructed text

    Returns:
        List of (start_char, end_char, entity_type, entity_text)
    """
    gold_entities = []
    for ann in ner_annotations:
        start_idx, end_idx, entity_type = ann[0], ann[1], ann[2]
        start_char, end_char, entity_text = token_indices_to_char_span(
            start_idx, end_idx, offsets, text
        )
        if start_char >= 0:
            gold_entities.append((start_char, end_char, entity_type, entity_text))

    return sorted(gold_entities, key=lambda x: (x[0], x[1]))


# =============================================================================
# Metrics calculation
# =============================================================================

def compute_metrics(
    gold_entities: List[Tuple[int, int, str, str]],
    pred_entities: List[Tuple[int, int, str, str]],
    entity_types: List[str]
) -> Dict[str, Any]:
    """
    Compute NER metrics (TP, FP, FN, P, R, F1) per class and aggregated.

    Entity matching: exact span match (start, end) AND same entity type.
    """
    # Per-class counts
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)

    # Create sets for matching
    gold_set = set()
    for start, end, etype, _ in gold_entities:
        gold_set.add((start, end, etype))

    pred_set = set()
    for start, end, etype, _ in pred_entities:
        pred_set.add((start, end, etype))

    # Count TP, FP, FN
    for start, end, etype in pred_set:
        if (start, end, etype) in gold_set:
            tp_per_class[etype] += 1
        else:
            fp_per_class[etype] += 1

    for start, end, etype in gold_set:
        if (start, end, etype) not in pred_set:
            fn_per_class[etype] += 1

    # Compute per-class metrics
    per_class = {}
    for etype in entity_types:
        tp = tp_per_class[etype]
        fp = fp_per_class[etype]
        fn = fn_per_class[etype]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[etype] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn  # gold count
        }

    # Micro-averaged metrics (sum all TP, FP, FN)
    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro-averaged metrics (average of per-class metrics, only for classes with support > 0)
    valid_classes = [etype for etype in entity_types if per_class[etype]["support"] > 0]
    if valid_classes:
        macro_precision = sum(per_class[e]["precision"] for e in valid_classes) / len(valid_classes)
        macro_recall = sum(per_class[e]["recall"] for e in valid_classes) / len(valid_classes)
        macro_f1 = sum(per_class[e]["f1"] for e in valid_classes) / len(valid_classes)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    return {
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        },
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4)
        },
        "per_class": per_class
    }


def print_metrics(metrics: Dict[str, Any], entity_types: List[str]):
    """Print metrics to console in a formatted table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Per-class metrics table
    print(f"\n{'Entity Type':<20} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'Support':>8}")
    print("-" * 80)

    per_class = metrics["per_class"]
    for etype in sorted(entity_types):
        m = per_class.get(etype, {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0, "support": 0})
        if m["support"] > 0:  # Only show classes with gold entities
            print(f"{etype:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                  f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['support']:>8}")

    print("-" * 80)

    # Aggregated metrics
    micro = metrics["micro"]
    macro = metrics["macro"]

    print(f"{'Micro-avg':<20} {micro['precision']:>8.4f} {micro['recall']:>8.4f} {micro['f1']:>8.4f} "
          f"{micro['tp']:>6} {micro['fp']:>6} {micro['fn']:>6}")
    print(f"{'Macro-avg':<20} {macro['precision']:>8.4f} {macro['recall']:>8.4f} {macro['f1']:>8.4f}")
    print("=" * 80)


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate_on_dataset(
    input_path: str,
    output_path: Optional[str],
    metrics_output: str,
    model_path: str,
    threshold: float,
    batch_size: int,
    measure_time: bool,
    timing_output: Optional[str],
    device: str
):
    """
    Run evaluation on GLiNER-format dataset.
    """
    timing = TimingStats() if measure_time else None
    if timing:
        timing.start()

    # Load dataset
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='UTF-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} documents")

    # Initialize inference
    print(f"\nInitializing GLiNERInference...")
    if timing:
        with timing.measure("model_loading"):
            inference = GLiNERInference(
                model_path=model_path,
                threshold=threshold,
                device=device
            )
            # Force model loading
            _ = inference.model
    else:
        inference = GLiNERInference(
            model_path=model_path,
            threshold=threshold,
            device=device
        )
        _ = inference.model

    entity_types = inference.get_entity_types()

    # Process documents
    all_predictions = []
    all_gold = []
    all_pred = []
    doc_times = []

    print(f"\nRunning inference on {len(dataset)} documents...")

    for doc_idx, doc in enumerate(tqdm(dataset, desc="Inference")):
        tokens = doc["tokenized_text"]
        gold_ner = doc.get("ner", [])

        # Convert tokens to text
        text, offsets = tokens_to_text_with_offsets(tokens)

        # Convert gold entities
        gold_entities = convert_gold_entities(gold_ner, offsets, text)

        # Run inference
        if timing:
            start_t = time.perf_counter()
            pred_entities = inference.predict(text)
            doc_times.append(time.perf_counter() - start_t)
        else:
            pred_entities = inference.predict(text)

        # Collect for metrics
        all_gold.extend(gold_entities)
        all_pred.extend(pred_entities)

        # Store predictions
        all_predictions.append({
            "doc_id": doc_idx,
            "text": text,
            "predictions": [list(e) for e in pred_entities],
            "gold": [list(e) for e in gold_entities]
        })

    # Record inference times
    if timing and doc_times:
        timing.times["inference_per_doc"] = doc_times

    # Compute metrics
    print("\nComputing metrics...")
    if timing:
        with timing.measure("metrics_calculation"):
            metrics = compute_metrics(all_gold, all_pred, entity_types)
    else:
        metrics = compute_metrics(all_gold, all_pred, entity_types)

    # Add summary
    metrics["summary"] = {
        "num_documents": len(dataset),
        "gold_entities": len(all_gold),
        "predicted_entities": len(all_pred),
        "model": model_path,
        "threshold": threshold
    }

    # Print metrics
    print_metrics(metrics, entity_types)

    print(f"\nSummary:")
    print(f"  Documents: {metrics['summary']['num_documents']}")
    print(f"  Gold entities: {metrics['summary']['gold_entities']}")
    print(f"  Predicted entities: {metrics['summary']['predicted_entities']}")

    # Save predictions
    if output_path:
        print(f"\nSaving predictions to {output_path}...")
        with open(output_path, 'w', encoding='UTF-8') as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    # Save metrics
    print(f"Saving metrics to {metrics_output}...")
    with open(metrics_output, 'w', encoding='UTF-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save timing
    if timing and timing_output:
        timing_stats = timing.get_stats()
        timing_stats["num_documents"] = len(dataset)
        if doc_times:
            timing_stats["throughput_docs_per_sec"] = round(
                len(dataset) / timing_stats.get("inference_per_doc_total_sec", 1), 2
            )

        print(f"Saving timing to {timing_output}...")
        with open(timing_output, 'w', encoding='UTF-8') as f:
            json.dump(timing_stats, f, ensure_ascii=False, indent=2)

        print(f"\nTiming statistics:")
        for key, value in timing_stats.items():
            print(f"  {key}: {value}")

    print("\nDone!")
    return metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GLiNER Inference and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on test set
  python gliner_inference.py --input test.json --metrics_output metrics.json

  # With timing and predictions output
  python gliner_inference.py --input test.json --output predictions.json \\
      --measure_time --timing_output timing.json

  # Custom model and threshold
  python gliner_inference.py --input test.json --model ./my_model --threshold 0.6
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Path to GLiNER-format JSON file (with tokenized_text and ner fields)"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Path to save predictions JSON (optional)"
    )
    parser.add_argument(
        '--metrics_output',
        type=str,
        default="metrics.json",
        help="Path to save metrics JSON (default: metrics.json)"
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default="urchade/gliner_multi-v2.1",
        help="Model path or HuggingFace ID (default: urchade/gliner_multi-v2.1)"
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help="Prediction confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=8,
        help="Batch size for inference (default: 8, currently unused)"
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda)"
    )
    parser.add_argument(
        '--measure_time',
        action='store_true',
        help="Enable timing measurements"
    )
    parser.add_argument(
        '--timing_output',
        type=str,
        default="timing.json",
        help="Path to save timing results JSON (default: timing.json)"
    )

    args = parser.parse_args()

    evaluate_on_dataset(
        input_path=args.input,
        output_path=args.output,
        metrics_output=args.metrics_output,
        model_path=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        measure_time=args.measure_time,
        timing_output=args.timing_output if args.measure_time else None,
        device=args.device
    )


if __name__ == "__main__":
    main()
