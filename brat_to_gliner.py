#!/usr/bin/env python3
"""
BRAT to GLiNER format converter.

Converts BRAT annotation format to GLiNER training JSON format.

BRAT format:
    file.txt - raw text
    file.ann - annotations: T1\tENTITY_TYPE start_char end_char\tentity_text

GLiNER format:
    {
        "tokenized_text": ["word1", "word2", ...],
        "ner": [[start_idx, end_idx, "ENTITY_TYPE"], ...]  # token indices, inclusive
    }

Usage:
    python brat_to_gliner.py \
        --brat_path /path/to/NEREL1.1 \
        --output_path /path/to/output \
        --labels_path ./conf/nerel_labels.json
"""

import json
import os
import argparse
from typing import List, Tuple, Dict, Any

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer
from tqdm.auto import tqdm

# Download NLTK data if needed
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def load_tokenizers():
    """Load Russian sentence and word tokenizers."""
    try:
        ru_tokenizer = load("tokenizers/punkt/russian.pickle")
    except LookupError:
        # Fallback to default punkt tokenizer
        ru_tokenizer = load("tokenizers/punkt/english.pickle")
        print("Warning: Russian tokenizer not found, using English")

    word_tokenizer = NLTKWordTokenizer()
    return ru_tokenizer, word_tokenizer


def tokenize_text(text: str, ru_tokenizer, word_tokenizer) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Tokenize text into words with character offsets.

    Returns:
        tokens: List of word strings
        offsets: List of (start_char, end_char) tuples for each token
    """
    offset_mapping = []

    # Sentence tokenization
    sentence_spans = list(ru_tokenizer.span_tokenize(text))

    for start, end in sentence_spans:
        context = text[start:end]
        # Word tokenization within sentence
        word_spans = list(word_tokenizer.span_tokenize(context))
        # Adjust offsets to document level
        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

    # Extract tokens from offsets
    tokens = [text[start:end] for start, end in offset_mapping]

    return tokens, offset_mapping


def parse_brat_annotations(ann_path: str, tags: List[str]) -> List[Tuple[str, int, int]]:
    """
    Parse BRAT annotation file.

    Returns:
        List of (entity_type, start_char, end_char) tuples
    """
    entities = []

    with open(ann_path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            # Parse entity annotation: T1\tTYPE start end\ttext
            if parts[0].startswith('T'):
                type_info = parts[1].split()
                if len(type_info) >= 3:
                    entity_type = type_info[0]

                    # Only include entities with allowed tags
                    if entity_type not in tags:
                        continue

                    try:
                        start_char = int(type_info[1])
                        end_char = int(type_info[2])
                        entities.append((entity_type, start_char, end_char))
                    except ValueError:
                        continue

    return entities


def char_to_token_indices(
    entity_start_char: int,
    entity_end_char: int,
    offsets: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Convert character offsets to token indices (inclusive).

    Returns:
        (start_token_idx, end_token_idx) - both inclusive
        Returns (-1, -1) if entity doesn't align with tokens
    """
    start_token = -1
    end_token = -1

    for idx, (tok_start, tok_end) in enumerate(offsets):
        # Find token containing entity start
        if start_token == -1 and tok_start <= entity_start_char < tok_end:
            start_token = idx
        # Also check if entity start exactly matches token start
        if start_token == -1 and tok_start == entity_start_char:
            start_token = idx

        # Find token containing entity end (end_char is exclusive in BRAT)
        # So we look for token that ends at or after entity_end_char
        if tok_start < entity_end_char <= tok_end:
            end_token = idx
        # Also check if token ends exactly at entity end
        if tok_end == entity_end_char:
            end_token = idx

    return start_token, end_token


def convert_document(
    txt_path: str,
    ann_path: str,
    tags: List[str],
    ru_tokenizer,
    word_tokenizer
) -> Dict[str, Any]:
    """
    Convert a single BRAT document to GLiNER format.

    Returns:
        GLiNER format dict or None if conversion fails
    """
    # Read text
    with open(txt_path, 'r', encoding='UTF-8') as f:
        text = f.read()

    if not text.strip():
        return None

    # Tokenize
    tokens, offsets = tokenize_text(text, ru_tokenizer, word_tokenizer)

    if not tokens:
        return None

    # Parse annotations
    entities = parse_brat_annotations(ann_path, tags)

    # Convert to token indices
    ner_annotations = []
    for entity_type, start_char, end_char in entities:
        start_idx, end_idx = char_to_token_indices(start_char, end_char, offsets)

        if start_idx >= 0 and end_idx >= 0 and start_idx <= end_idx:
            ner_annotations.append([start_idx, end_idx, entity_type])

    # Sort by position
    ner_annotations.sort(key=lambda x: (x[0], x[1]))

    return {
        "tokenized_text": tokens,
        "ner": ner_annotations
    }


def convert_dataset(
    brat_path: str,
    output_path: str,
    labels_path: str,
    splits: List[str] = ["train", "dev", "test"]
):
    """
    Convert BRAT dataset to GLiNER format.
    """
    # Load labels
    with open(labels_path, 'r', encoding='UTF-8') as f:
        labels_config = json.load(f)

    tags = labels_config.get("labels", [])
    print(f"Loaded {len(tags)} entity types: {tags[:5]}...")

    # Load tokenizers
    ru_tokenizer, word_tokenizer = load_tokenizers()

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    for split in splits:
        split_path = os.path.join(brat_path, split)

        if not os.path.exists(split_path):
            print(f"Warning: Split '{split}' not found at {split_path}, skipping...")
            continue

        print(f"\nProcessing {split} split...")

        # Find all annotation files
        ann_files = []
        for root, dirs, files in os.walk(split_path):
            for f in files:
                if f.endswith('.ann'):
                    ann_files.append(os.path.join(root, f))

        # Convert documents
        documents = []
        entity_count = 0
        skipped = 0

        for ann_path in tqdm(ann_files, desc=f"Converting {split}"):
            txt_path = ann_path[:-4] + '.txt'

            if not os.path.exists(txt_path):
                skipped += 1
                continue

            # Skip empty files
            if os.path.getsize(ann_path) == 0:
                skipped += 1
                continue

            try:
                doc = convert_document(txt_path, ann_path, tags, ru_tokenizer, word_tokenizer)
                if doc:
                    documents.append(doc)
                    entity_count += len(doc["ner"])
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing {ann_path}: {e}")
                skipped += 1

        # Write output
        output_file = os.path.join(output_path, f"{split}.json")
        with open(output_file, 'w', encoding='UTF-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"  Documents: {len(documents)}")
        print(f"  Entities: {entity_count}")
        print(f"  Skipped: {skipped}")
        print(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert BRAT format to GLiNER training format"
    )
    parser.add_argument(
        '--brat_path',
        type=str,
        required=True,
        help="Path to BRAT dataset (with train, dev, test subdirs)"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path for GLiNER format output"
    )
    parser.add_argument(
        '--labels_path',
        type=str,
        default='./conf/nerel_labels.json',
        help="Path to labels JSON file"
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'dev', 'test'],
        help="Dataset splits to convert"
    )

    args = parser.parse_args()

    convert_dataset(
        brat_path=args.brat_path,
        output_path=args.output_path,
        labels_path=args.labels_path,
        splits=args.splits
    )

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
