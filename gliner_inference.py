#!/usr/bin/env python3
"""
GLiNER Inference Module.

Provides GLiNERInference class with the same interface as BinderInference
for named entity recognition on Russian text.

Usage (as library - direct parameters):
    from gliner_inference import GLiNERInference

    inference = GLiNERInference(model_path="urchade/gliner_multi-v2.1")
    entities = inference.predict("Владимир Путин посетил Москву 5 января 2024 года.")
    # Returns: [(0, 14, 'PERSON', 'Владимир Путин'), (24, 30, 'CITY', 'Москву'), ...]

Usage (as library - folder-based, like BinderInference):
    from gliner_inference import GLiNERInference, download_model_to_folder

    # Step 1: Download model for offline use (run once, with internet)
    download_model_to_folder("urchade/gliner_multi-v2.1", "./my_model")

    # Step 2: Use offline (no internet needed)
    inference = GLiNERInference(config_path="./my_model")
    entities = inference.predict("Владимир Путин посетил Москву.")

Usage (CLI evaluation):
    python gliner_inference.py --input test.json --output predictions.json --metrics_output metrics.json
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
from gliner import GLiNER
from gliner.infer_packing import InferencePackingConfig
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

    Supports two initialization modes:
    1. Folder-based (like BinderInference): config_path points to a folder
       containing config.json and model files for offline use.
    2. Direct parameters (backward compatible): specify model_path, labels, etc.

    Sliding Window Support:
        For texts longer than max_tokens, the text is split into overlapping
        windows and processed separately. Results are merged with deduplication.
        Configure via max_tokens and stride_tokens parameters.
    """

    def __init__(
        self,
        # New: folder-based initialization (like BinderInference)
        config_path: Optional[str] = None,
        # Existing: direct parameters (backward compatible)
        model_path: Optional[str] = None,
        device: str = "auto",
        threshold: float = 0.5,
        labels: Optional[List[str]] = None,
        labels_path: Optional[str] = None,
        flat_ner: bool = True,
        # New: match BinderInference parameter
        prediction_threshold_factor: float = 1.0,
        # Sliding window parameters (like Binder's doc_stride)
        max_tokens: int = 384,
        stride_tokens: int = 128,
    ):
        """
        Initialize GLiNER inference.

        Two initialization modes:

        1. Folder-based (config_path):
           ```
           inference = GLiNERInference(config_path="./my_model_folder")
           ```
           The folder should contain config.json with model settings.

        2. Direct parameters (backward compatible):
           ```
           inference = GLiNERInference(model_path="urchade/gliner_multi-v2.1")
           ```

        Args:
            config_path: Path to folder containing config.json and model files.
                        If provided, other parameters are loaded from config.
            model_path: Path to GLiNER model or HuggingFace model ID.
                       Default: "urchade/gliner_multi-v2.1"
            device: Device to use ("auto", "cpu", "cuda", "cuda:0", etc.)
            threshold: Prediction confidence threshold (0.0-1.0)
            labels: List of entity labels to predict.
                   Default: NEREL 29 entity types
            labels_path: Path to JSON file with labels config
            flat_ner: If True, no overlapping entities. If False, allow nested.
            prediction_threshold_factor: Multiplier for threshold (for compatibility
                                        with BinderInference). Final threshold =
                                        threshold * prediction_threshold_factor.
            max_tokens: Maximum number of word tokens per window (default: 384).
                       Texts longer than this are split into overlapping windows.
                       Similar to Binder's max_seq_length.
            stride_tokens: Overlap between consecutive windows in tokens (default: 128).
                          Similar to Binder's doc_stride. Entities in overlap regions
                          are deduplicated.
        """
        # Store config path for reference
        self._config_path = config_path
        self._config_dir: Optional[str] = None
        self._batch_mode: bool = False

        # If config_path provided, load settings from config file
        if config_path is not None:
            config = self._load_config(config_path)
            self._config_dir = os.path.dirname(config_path) if config_path.endswith('.json') else config_path

            # Extract settings from config (with fallbacks to direct parameters)
            model_path = config.get("model_path", model_path)
            device = config.get("device", device)
            threshold = config.get("threshold", threshold)
            flat_ner = config.get("flat_ner", flat_ner)
            # Sliding window parameters
            max_tokens = config.get("max_tokens", max_tokens)
            stride_tokens = config.get("stride_tokens", stride_tokens)

            # Handle relative model_path ("." means same folder as config)
            if model_path == "." or model_path == "./":
                model_path = self._config_dir
            elif model_path and not os.path.isabs(model_path) and not model_path.startswith((".", "/")):
                # Check if it's a HuggingFace ID (contains / but no path separators)
                if "/" in model_path and not os.path.exists(model_path):
                    # Likely a HuggingFace model ID like "urchade/gliner_multi-v2.1"
                    pass
                elif os.path.exists(os.path.join(self._config_dir, model_path)):
                    # Relative path from config directory
                    model_path = os.path.join(self._config_dir, model_path)

            # Load labels from config
            labels = self._load_labels_from_config(config, self._config_dir, labels)

        self.model_path = model_path or "urchade/gliner_multi-v2.1"
        self.base_threshold = threshold
        self.prediction_threshold_factor = prediction_threshold_factor
        self.threshold = threshold * prediction_threshold_factor
        self.flat_ner = flat_ner

        # Sliding window configuration
        self.max_tokens = max_tokens
        self.stride_tokens = stride_tokens

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load labels (only if not already loaded from config)
        if labels is not None:
            self.labels = labels
        else:
            self.labels = self._load_labels(None, labels_path)

        # Lazy model loading
        self._model: Optional[GLiNER] = None

        # Inference optimization: packing config for batching short sequences
        self._packing_config = InferencePackingConfig(max_length=512)

        # Inference optimization: cache label embeddings for BiEncoder models
        # Maps frozenset of labels -> (labels_list, embeddings_tensor)
        self._label_embeddings_cache: Dict[frozenset, Tuple[List[str], torch.Tensor]] = {}

    def _load_config(self, config_path: str) -> dict:
        """
        Load config from folder or config.json file.

        Args:
            config_path: Path to folder containing config.json, or path to config.json directly

        Returns:
            Config dictionary
        """
        # Determine the config file path
        if os.path.isdir(config_path):
            config_file = os.path.join(config_path, "config.json")
        else:
            config_file = config_path

        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Config file not found: {config_file}. "
                f"Expected a config.json file in the specified folder."
            )

        with open(config_file, 'r', encoding='UTF-8') as f:
            config = json.load(f)

        return config

    def _load_labels_from_config(
        self,
        config: dict,
        config_dir: str,
        fallback_labels: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """
        Load labels from config (inline or file reference).

        Args:
            config: Config dictionary
            config_dir: Directory containing the config file
            fallback_labels: Labels to use if not found in config

        Returns:
            List of labels, or None to use default loading
        """
        # Option 1: Inline labels in config
        if "labels" in config and isinstance(config["labels"], list):
            return config["labels"]

        # Option 2: Labels file reference
        if "labels_file" in config:
            labels_file = config["labels_file"]
            # Handle relative path
            if not os.path.isabs(labels_file):
                labels_file = os.path.join(config_dir, labels_file)

            if os.path.exists(labels_file):
                with open(labels_file, 'r', encoding='UTF-8') as f:
                    labels_config = json.load(f)
                    if isinstance(labels_config, list):
                        return labels_config
                    elif "labels" in labels_config:
                        return labels_config["labels"]

        return fallback_labels

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

    def _is_biencoder(self) -> bool:
        """Check if the model is a BiEncoder (has encode_labels method)."""
        return hasattr(self.model, 'encode_labels')

    def _get_cached_label_embeddings(
        self, labels: List[str]
    ) -> Optional[Tuple[List[str], torch.Tensor]]:
        """Get or compute cached label embeddings for BiEncoder models.

        Args:
            labels: List of entity labels to get embeddings for

        Returns:
            Tuple of (labels_list, embeddings_tensor) if BiEncoder, None otherwise
        """
        if not self._is_biencoder():
            return None

        labels_key = frozenset(labels)

        if labels_key not in self._label_embeddings_cache:
            # Compute and cache embeddings
            embeddings = self.model.encode_labels(labels)
            self._label_embeddings_cache[labels_key] = (list(labels), embeddings)

        return self._label_embeddings_cache[labels_key]

    _WHITESPACE_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def _tokenize_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize text into words with character offsets.

        Uses the same regex as GLiNER's WhitespaceTokenSplitter so that
        token counts match exactly, preventing silent truncation in the
        model processor.

        Returns:
            List of (token, start_char, end_char) tuples
        """
        return [(m.group(), m.start(), m.end()) for m in self._WHITESPACE_PATTERN.finditer(text)]

    def _create_windows(
        self,
        tokens: List[Tuple[str, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Split tokens into overlapping windows.

        Args:
            tokens: List of (token, start_char, end_char) tuples

        Returns:
            List of (token_start_idx, token_end_idx, char_start, char_end) tuples
            Each tuple defines a window of tokens to process.
        """
        if len(tokens) <= self.max_tokens:
            # Text fits in single window
            if not tokens:
                return []
            return [(0, len(tokens), tokens[0][1], tokens[-1][2])]

        windows = []
        step = self.max_tokens - self.stride_tokens
        if step <= 0:
            step = self.max_tokens // 2  # Fallback if stride >= max_tokens

        for start_idx in range(0, len(tokens), step):
            end_idx = min(start_idx + self.max_tokens, len(tokens))
            char_start = tokens[start_idx][1]
            char_end = tokens[end_idx - 1][2]
            windows.append((start_idx, end_idx, char_start, char_end))

            if end_idx >= len(tokens):
                break

        return windows

    def _merge_entities(
        self,
        all_entities: List[Tuple[int, int, str, str]]
    ) -> List[Tuple[int, int, str, str]]:
        """
        Merge entities from multiple windows, removing duplicates.

        Entities are considered duplicates if they have the same
        (start, end, type). In case of duplicates, keep one.

        Args:
            all_entities: List of (start_char, end_char, entity_type, entity_text)

        Returns:
            Deduplicated and sorted list of entities
        """
        # Use set for deduplication (based on position and type)
        seen = set()
        unique_entities = []

        for entity in all_entities:
            key = (entity[0], entity[1], entity[2])  # start, end, type
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        # Sort by position
        unique_entities.sort(key=lambda x: (x[0], x[1]))
        return unique_entities

    def _predict_single_window(
        self,
        text: str,
        char_offset: int = 0
    ) -> List[Tuple[int, int, str, str]]:
        """
        Run prediction on a single text window (no sliding window).

        Args:
            text: Text to process (should fit within max_tokens)
            char_offset: Offset to add to character positions

        Returns:
            List of (start_char, end_char, entity_type, entity_text)
        """
        if not text or not text.strip():
            return []

        # Use cached label embeddings for BiEncoder models
        cached = self._get_cached_label_embeddings(self.labels)

        if cached is not None:
            labels_list, labels_embeddings = cached
            entities = self.model.predict_with_embeds(
                text,
                labels_embeddings,
                labels_list,
                threshold=self.threshold,
                flat_ner=self.flat_ner
            )
        else:
            # Predict entities using GLiNER (UniEncoder path)
            entities = self.model.predict_entities(
                text,
                self.labels,
                threshold=self.threshold,
                flat_ner=self.flat_ner
            )

        # Convert to BinderInference format with offset adjustment
        results = []
        for entity in entities:
            start_char = entity['start'] + char_offset
            end_char = entity['end'] + char_offset
            entity_type = entity['label']
            entity_text = entity['text']
            results.append((start_char, end_char, entity_type, entity_text))

        return results

    def predict(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Perform named entity recognition on input text.

        For texts longer than max_tokens, uses sliding window approach
        to process the entire text without truncation.

        Args:
            text: Input text string

        Returns:
            List of (start_char, end_char, entity_type, entity_text) tuples,
            sorted by position.
        """
        if not text or not text.strip():
            return []

        # Tokenize text to determine if sliding window is needed
        tokens = self._tokenize_text(text)

        if len(tokens) <= self.max_tokens:
            # Text fits in single window - use direct prediction
            return self._predict_single_window(text, char_offset=0)

        # Text is too long - use sliding window approach
        windows = self._create_windows(tokens)
        all_entities = []

        for token_start_idx, token_end_idx, char_start, char_end in windows:
            # Extract window text
            window_text = text[char_start:char_end]

            # Predict on window
            window_entities = self._predict_single_window(
                window_text,
                char_offset=char_start
            )
            all_entities.extend(window_entities)

        # Merge and deduplicate entities from all windows
        results = self._merge_entities(all_entities)

        return results

    def _predict_windows_batch(
        self,
        window_texts: List[str],
        char_offsets: List[int]
    ) -> List[List[Tuple[int, int, str, str]]]:
        """
        Process multiple window texts in an optimized batch.

        Args:
            window_texts: List of text windows to process
            char_offsets: Character offset for each window (to adjust entity positions)

        Returns:
            List of entity lists, one per window
        """
        if not window_texts:
            return []

        # Use cached label embeddings for BiEncoder models
        cached = self._get_cached_label_embeddings(self.labels)

        if cached is not None:
            labels_list, labels_embeddings = cached
            all_entities = self.model.batch_predict_with_embeds(
                window_texts,
                labels_embeddings,
                labels_list,
                threshold=self.threshold,
                flat_ner=self.flat_ner,
                packing_config=self._packing_config
            )
        else:
            all_entities = self.model.inference(
                window_texts,
                self.labels,
                threshold=self.threshold,
                flat_ner=self.flat_ner,
                packing_config=self._packing_config
            )

        # Convert to BinderInference format with offset adjustment
        results = []
        for window_entities, char_offset in zip(all_entities, char_offsets):
            window_results = []
            for entity in window_entities:
                start_char = entity['start'] + char_offset
                end_char = entity['end'] + char_offset
                entity_type = entity['label']
                entity_text = entity['text']
                window_results.append((start_char, end_char, entity_type, entity_text))
            results.append(window_results)

        return results

    def predict_batch(
        self,
        texts: Union[List[str], Tuple[str, ...], str]
    ) -> List[List[Tuple[int, int, str, str]]]:
        """
        Perform batch prediction on multiple texts.

        Supports sliding window for long texts - all windows from all texts
        are collected into a single optimized batch for efficient processing.

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

        # Prepare windows for all texts
        # Structure: list of (text_idx, window_text, char_offset)
        all_windows = []
        text_window_counts = []  # How many windows per text

        for text_idx, text in enumerate(texts):
            if not text or not text.strip():
                text_window_counts.append(0)
                continue

            tokens = self._tokenize_text(text)

            if len(tokens) <= self.max_tokens:
                # Single window for this text
                all_windows.append((text_idx, text, 0))
                text_window_counts.append(1)
            else:
                # Multiple windows needed
                windows = self._create_windows(tokens)
                for token_start_idx, token_end_idx, char_start, char_end in windows:
                    window_text = text[char_start:char_end]
                    all_windows.append((text_idx, window_text, char_start))
                text_window_counts.append(len(windows))

        if not all_windows:
            return [[] for _ in texts]

        # Extract window texts and offsets for batch processing
        window_texts = [w[1] for w in all_windows]
        char_offsets = [w[2] for w in all_windows]

        # Process all windows in one optimized batch
        window_results = self._predict_windows_batch(window_texts, char_offsets)

        # Reassemble results back to original texts
        results = [[] for _ in texts]
        window_idx = 0

        for text_idx, window_count in enumerate(text_window_counts):
            if window_count == 0:
                continue

            # Collect all entities from windows of this text
            text_entities = []
            for _ in range(window_count):
                text_entities.extend(window_results[window_idx])
                window_idx += 1

            # Merge and deduplicate if multiple windows
            if window_count > 1:
                text_entities = self._merge_entities(text_entities)
            else:
                text_entities.sort(key=lambda x: (x[0], x[1]))

            results[text_idx] = text_entities

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
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "threshold": self.threshold,
            "base_threshold": self.base_threshold,
            "prediction_threshold_factor": self.prediction_threshold_factor,
            "flat_ner": self.flat_ner,
            "entity_types": self.labels,
            "num_entity_types": len(self.labels),
            "model_loaded": self._model is not None,
            "batch_mode": self._batch_mode,
            # Sliding window parameters
            "max_tokens": self.max_tokens,
            "stride_tokens": self.stride_tokens,
        }
        if self._config_path:
            info["config_path"] = self._config_path
            info["config_dir"] = self._config_dir
        return info

    def set_threshold(self, threshold: float) -> None:
        """Update prediction threshold."""
        self.threshold = threshold

    def set_labels(self, labels: List[str]) -> None:
        """Update entity labels to predict."""
        self.labels = labels

    def set_sliding_window(self, max_tokens: int, stride_tokens: int) -> None:
        """
        Update sliding window parameters.

        Args:
            max_tokens: Maximum tokens per window
            stride_tokens: Overlap between consecutive windows
        """
        if stride_tokens >= max_tokens:
            raise ValueError(f"stride_tokens ({stride_tokens}) must be less than max_tokens ({max_tokens})")
        self.max_tokens = max_tokens
        self.stride_tokens = stride_tokens

    def clear_cache(self) -> None:
        """
        Clear internal caches (for BinderInference compatibility).

        Clears the label embeddings cache and GPU memory if applicable.
        """
        # Clear label embeddings cache
        self._label_embeddings_cache.clear()

        if hasattr(self, '_model') and self._model is not None:
            # Clear CUDA cache if using GPU
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

    def enable_batch_mode(self, batch_size: int = 8) -> None:
        """
        Enable batch optimization mode (for BinderInference compatibility).

        Args:
            batch_size: Default batch size for batch operations
        """
        self._batch_mode = True
        self._default_batch_size = batch_size

    def disable_batch_mode(self) -> None:
        """Disable batch optimization mode."""
        self._batch_mode = False


# =============================================================================
# Model download helper
# =============================================================================

def download_model_to_folder(
    model_id: str,
    output_folder: str,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    flat_ner: bool = True,
    device: str = "auto",
    max_tokens: int = 384,
    stride_tokens: int = 128,
) -> str:
    """
    Download a GLiNER model from HuggingFace and create a config.json for offline use.

    This function:
    1. Downloads the model and tokenizer files from HuggingFace
    2. Creates a config.json file with inference settings
    3. Optionally creates a labels.json file

    After running this, you can use the folder offline:
        inference = GLiNERInference(config_path=output_folder)

    Args:
        model_id: HuggingFace model ID (e.g., "urchade/gliner_multi-v2.1")
        output_folder: Directory to save the model files
        labels: Entity labels to include in config. If None, uses NEREL defaults.
        threshold: Default prediction threshold
        flat_ner: Default flat_ner setting
        device: Default device setting

    Returns:
        Path to the output folder

    Example:
        # Download model for offline use
        download_model_to_folder(
            "urchade/gliner_multi-v2.1",
            "./my_offline_model",
            labels=["PERSON", "ORGANIZATION", "LOCATION"]
        )

        # Later, use offline (no internet needed)
        inference = GLiNERInference(config_path="./my_offline_model")
    """
    from huggingface_hub import snapshot_download

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    print(f"Downloading model {model_id} to {output_folder}...")

    # Download model files from HuggingFace
    # Skip safetensors if pytorch_model.bin exists (avoids downloading 2x model weights)
    snapshot_download(
        repo_id=model_id,
        local_dir=output_folder,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors"]
    )

    # Use provided labels or defaults
    if labels is None:
        labels = DEFAULT_LABELS

    # Create config.json for inference
    config = {
        "model_path": ".",
        "threshold": threshold,
        "flat_ner": flat_ner,
        "device": device,
        "labels": labels,
        # Sliding window parameters
        "max_tokens": max_tokens,
        "stride_tokens": stride_tokens,
        "_source_model": model_id,
        "_created_by": "download_model_to_folder"
    }

    config_path = os.path.join(output_folder, "config.json")
    with open(config_path, 'w', encoding='UTF-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Model downloaded successfully to {output_folder}")
    print(f"Config file created: {config_path}")
    print(f"\nUsage:")
    print(f'  inference = GLiNERInference(config_path="{output_folder}")')

    return output_folder


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


# =============================================================================
# BRAT Format Support
# =============================================================================

def load_brat_folder(
    folder_path: str,
    labels: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Load documents from a BRAT-format folder.

    Args:
        folder_path: Path to folder containing .txt and .ann files
        labels: Entity labels to include. If None, includes all.

    Returns:
        List of dicts with 'text', 'entities' (as char offsets), 'doc_id'
    """
    import glob

    if labels is None:
        labels = DEFAULT_LABELS

    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    documents = []

    for txt_path in txt_files:
        doc_id = os.path.splitext(os.path.basename(txt_path))[0]
        ann_path = txt_path.replace(".txt", ".ann")

        # Read text
        with open(txt_path, 'r', encoding='UTF-8') as f:
            text = f.read()

        # Parse annotations
        entities = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='UTF-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('T'):
                        continue

                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue

                    type_info = parts[1].split()
                    if len(type_info) >= 3:
                        entity_type = type_info[0]
                        if entity_type not in labels:
                            continue
                        try:
                            start = int(type_info[1])
                            end = int(type_info[2])
                            entity_text = parts[2] if len(parts) > 2 else text[start:end]
                            entities.append((start, end, entity_type, entity_text))
                        except ValueError:
                            continue

        documents.append({
            'doc_id': doc_id,
            'text': text,
            'entities': entities
        })

    return documents


def evaluate_on_brat(
    brat_folder: str,
    model_path: str,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Evaluate a model on BRAT-format test data.

    Args:
        brat_folder: Path to folder with .txt/.ann files
        model_path: Model path or HuggingFace ID
        labels: Entity labels to evaluate
        threshold: Prediction threshold
        device: Device to use

    Returns:
        Dict with metrics
    """
    if labels is None:
        labels = DEFAULT_LABELS

    # Load data
    print(f"Loading BRAT data from {brat_folder}...")
    documents = load_brat_folder(brat_folder, labels)
    print(f"Loaded {len(documents)} documents")

    # Initialize model
    print(f"Loading model from {model_path}...")
    inference = GLiNERInference(
        model_path=model_path,
        labels=labels,
        threshold=threshold,
        device=device
    )
    _ = inference.model  # Force load

    # Run predictions
    all_gold = []
    all_pred = []

    print("Running inference...")
    for doc in tqdm(documents, desc="Evaluating"):
        text = doc['text']
        gold_entities = doc['entities']

        # Predict
        pred_entities = inference.predict(text)

        all_gold.extend(gold_entities)
        all_pred.extend(pred_entities)

    # Compute metrics
    metrics = compute_metrics(all_gold, all_pred, labels)
    metrics['summary'] = {
        'num_documents': len(documents),
        'gold_entities': len(all_gold),
        'predicted_entities': len(all_pred),
        'model': model_path,
        'threshold': threshold
    }

    return metrics


def compare_models_on_brat(
    brat_folder: str,
    model_paths: List[str],
    model_names: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
    device: str = "auto",
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same BRAT test data.

    Args:
        brat_folder: Path to folder with .txt/.ann files
        model_paths: List of model paths to compare
        model_names: Optional names for each model
        labels: Entity labels to evaluate
        threshold: Prediction threshold
        device: Device to use
        output_path: Optional path to save comparison results

    Returns:
        Dict mapping model name to metrics
    """
    if labels is None:
        labels = DEFAULT_LABELS

    if model_names is None:
        model_names = [os.path.basename(p) for p in model_paths]

    # Load data once
    print(f"Loading BRAT data from {brat_folder}...")
    documents = load_brat_folder(brat_folder, labels)
    print(f"Loaded {len(documents)} documents\n")

    results = {}

    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Path: {model_path}")
        print('='*60)

        # Initialize model
        inference = GLiNERInference(
            model_path=model_path,
            labels=labels,
            threshold=threshold,
            device=device
        )
        _ = inference.model

        # Run predictions
        all_gold = []
        all_pred = []

        for doc in tqdm(documents, desc=f"Inference ({model_name})"):
            text = doc['text']
            gold_entities = doc['entities']
            pred_entities = inference.predict(text)

            all_gold.extend(gold_entities)
            all_pred.extend(pred_entities)

        # Compute metrics
        metrics = compute_metrics(all_gold, all_pred, labels)
        metrics['summary'] = {
            'num_documents': len(documents),
            'gold_entities': len(all_gold),
            'predicted_entities': len(all_pred),
            'model': model_path,
            'threshold': threshold
        }

        # Print summary
        print(f"\n{model_name} Results:")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  F1:        {metrics['overall']['f1']:.4f}")

        results[model_name] = metrics

        # Free memory
        del inference
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print comparison table
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print('='*60)
    print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print('-'*60)
    for name, metrics in results.items():
        p = metrics['overall']['precision']
        r = metrics['overall']['recall']
        f1 = metrics['overall']['f1']
        print(f"{name:<30} {p:>10.4f} {r:>10.4f} {f1:>10.4f}")
    print('='*60)

    # Save results
    if output_path:
        with open(output_path, 'w', encoding='UTF-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GLiNER Inference and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on test set (JSON format)
  python gliner_inference.py --input test.json --metrics_output metrics.json

  # Compare models on BRAT format data
  python gliner_inference.py --compare --brat_folder /path/to/test \\
      --models ./model1 ./model2 --names "Baseline" "Finetuned"

  # Custom model and threshold
  python gliner_inference.py --input test.json --model ./my_model --threshold 0.6
        """
    )

    # Mode selection
    parser.add_argument(
        '--compare',
        action='store_true',
        help="Compare multiple models on BRAT data"
    )

    # BRAT comparison mode arguments
    parser.add_argument(
        '--brat_folder',
        type=str,
        help="Path to BRAT format folder (for --compare mode)"
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help="Model paths to compare (for --compare mode)"
    )
    parser.add_argument(
        '--names',
        type=str,
        nargs='+',
        help="Names for models (for --compare mode)"
    )

    # JSON evaluation mode arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
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

    # Common arguments
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

    if args.compare:
        # Compare mode
        if not args.brat_folder:
            parser.error("--compare requires --brat_folder")
        if not args.models:
            parser.error("--compare requires --models")

        compare_models_on_brat(
            brat_folder=args.brat_folder,
            model_paths=args.models,
            model_names=args.names,
            threshold=args.threshold,
            device=args.device,
            output_path=args.metrics_output
        )
    else:
        # Standard JSON evaluation mode
        if not args.input:
            parser.error("--input is required for JSON evaluation mode")

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


def run_smoke_test():
    """
    Quick smoke test comparing pretrained vs finetuned models.

    Tests:
    1. Single text inference
    2. Batch inference (3 texts)

    This is the default action when running the script directly.
    """
    PRETRAINED_MODEL = "urchade/gliner_multi-v2.1"
    FINETUNED_MODEL = "./saved_ckpt/nerel-finetuned"

    # Test texts (Russian, from NEREL domain)
    SINGLE_TEXT = (
        "Владимир Путин встретился с президентом Франции Эммануэлем Макроном "
        "в Москве 7 февраля 2022 года для обсуждения ситуации на Украине."
    )

    BATCH_TEXTS = [
        "Компания Apple представила новый iPhone 15 в Купертино.",
        "Александр Пушкин родился в Москве в 1799 году и стал великим русским поэтом.",
        "Газпром подписал контракт с Китаем на поставку 38 миллиардов кубометров газа.",
    ]

    # Test labels (subset for cleaner output)
    TEST_LABELS = ["PERSON", "ORGANIZATION", "CITY", "COUNTRY", "DATE", "PRODUCT", "MONEY", "NUMBER"]

    print("="*70)
    print("GLiNER Smoke Test")
    print("="*70)

    models = [
        ("Pretrained", PRETRAINED_MODEL),
        ("Finetuned", FINETUNED_MODEL),
    ]

    for model_name, model_path in models:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"Path: {model_path}")
        print("="*70)

        try:
            inference = GLiNERInference(
                model_path=model_path,
                labels=TEST_LABELS,
                threshold=0.5,
                device="auto"
            )
            _ = inference.model  # Force load
        except Exception as e:
            print(f"ERROR loading model: {e}")
            continue

        # Test 1: Single text
        print(f"\n--- Single Text Inference ---")
        print(f"Text: {SINGLE_TEXT[:80]}...")
        entities = inference.predict(SINGLE_TEXT)
        print(f"Found {len(entities)} entities:")
        for start, end, label, text in entities:
            print(f"  [{label:12}] '{text}' ({start}-{end})")

        # Test 2: Batch inference
        print(f"\n--- Batch Inference ({len(BATCH_TEXTS)} texts) ---")
        batch_results = inference.predict_batch(BATCH_TEXTS)

        for i, (text, entities) in enumerate(zip(BATCH_TEXTS, batch_results)):
            print(f"\nText {i+1}: {text[:60]}...")
            print(f"Found {len(entities)} entities:")
            for start, end, label, ent_text in entities:
                print(f"  [{label:12}] '{ent_text}' ({start}-{end})")

        # Cleanup
        del inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("Smoke test complete!")
    print("="*70)


if __name__ == "__main__":
    import sys

    # If no arguments provided, run smoke test
    if len(sys.argv) == 1:
        run_smoke_test()
    else:
        main()
