#!/usr/bin/env python3
"""
Sliding-window preprocessor for GLiNER training data.

Splits long examples into overlapping windows so the model trains on inputs
within DeBERTa's effective 512-subword range, matching inference conditions.

Usage:
    python prepare_windowed_data.py \
        --input /path/to/train.json \
        --output /path/to/train_w384.json \
        --window_size 384 --stride 128
"""

import json
import argparse
from typing import List, Dict, Any, Tuple


def window_example(
    tokens: List[str],
    ner: List[List],
    window_size: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """
    Split a single example into overlapping windows.

    Only entities fully contained within a window are kept;
    their indices are adjusted relative to the window start.

    Args:
        tokens: Word-level token list.
        ner: Entity annotations as [start, end, label] (inclusive indices).
        window_size: Maximum tokens per window.
        stride: Overlap between consecutive windows.

    Returns:
        List of windowed examples (each with tokenized_text and ner).
    """
    n = len(tokens)
    if n <= window_size:
        return [{"tokenized_text": tokens, "ner": ner}]

    step = window_size - stride
    windows = []
    start = 0

    while start < n:
        end = min(start + window_size, n)
        window_tokens = tokens[start:end]

        # Keep only entities fully inside this window
        window_ner = []
        for ent in ner:
            ent_start, ent_end, label = ent[0], ent[1], ent[2]
            if ent_start >= start and ent_end < end:
                window_ner.append([ent_start - start, ent_end - start, label])

        windows.append({
            "tokenized_text": window_tokens,
            "ner": window_ner,
        })

        if end == n:
            break
        start += step

    return windows


def prepare_windowed_data(
    input_path: str,
    output_path: str,
    window_size: int = 384,
    stride: int = 128,
) -> None:
    """
    Read a GLiNER JSON dataset, split long examples into windows, and save.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total_in = len(data)
    total_entities_in = sum(len(d.get("ner", [])) for d in data)

    short_count = 0
    long_count = 0
    windows_from_long = 0
    out: List[Dict[str, Any]] = []

    for example in data:
        tokens = example["tokenized_text"]
        ner = example.get("ner", [])

        if len(tokens) <= window_size:
            short_count += 1
            out.append(example)
        else:
            long_count += 1
            wins = window_example(tokens, ner, window_size, stride)
            windows_from_long += len(wins)
            out.extend(wins)

    total_entities_out = sum(len(d.get("ner", [])) for d in out)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # Print statistics
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"  Examples in:  {total_in}")
    print(f"    Short (<= {window_size} tokens): {short_count} (passed through)")
    print(f"    Long  (>  {window_size} tokens): {long_count} -> {windows_from_long} windows")
    print(f"  Examples out: {len(out)}")
    print(f"  Entities in:  {total_entities_in}")
    print(f"  Entities out: {total_entities_out} "
          f"({total_entities_out - total_entities_in:+d} from overlap duplication)")


def main():
    parser = argparse.ArgumentParser(
        description="Split long GLiNER training examples into sliding windows"
    )
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--window_size", type=int, default=384,
        help="Max tokens per window (default: 384)"
    )
    parser.add_argument(
        "--stride", type=int, default=128,
        help="Overlap between consecutive windows (default: 128)"
    )
    args = parser.parse_args()

    if args.stride >= args.window_size:
        parser.error("stride must be less than window_size")

    prepare_windowed_data(args.input, args.output, args.window_size, args.stride)


if __name__ == "__main__":
    main()
