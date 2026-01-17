#!/usr/bin/env python3
"""
Convert NER metrics JSON files to CSV format.

Usage:
    python metrics_to_csv.py metrics.json                    # outputs to stdout
    python metrics_to_csv.py metrics.json -o metrics.csv     # outputs to file
    python metrics_to_csv.py m1.json m2.json -o compare.csv  # compare multiple
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_metrics(file_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def metrics_to_csv_single(metrics: Dict[str, Any], model_name: str = None) -> List[List[str]]:
    """Convert single metrics file to CSV rows."""
    rows = []

    # Header
    header = ["entity_type", "precision", "recall", "f1", "tp", "fp", "fn", "support"]
    if model_name:
        header.insert(0, "model")
    rows.append(header)

    # Per-class metrics
    per_class = metrics.get("per_class", {})
    for entity_type in sorted(per_class.keys()):
        data = per_class[entity_type]
        row = [
            entity_type,
            f"{data.get('precision', 0):.4f}",
            f"{data.get('recall', 0):.4f}",
            f"{data.get('f1', 0):.4f}",
            str(data.get('tp', 0)),
            str(data.get('fp', 0)),
            str(data.get('fn', 0)),
            str(data.get('support', 0))
        ]
        if model_name:
            row.insert(0, model_name)
        rows.append(row)

    # Micro average
    micro = metrics.get("micro", {})
    micro_row = [
        "MICRO_AVG",
        f"{micro.get('precision', 0):.4f}",
        f"{micro.get('recall', 0):.4f}",
        f"{micro.get('f1', 0):.4f}",
        str(micro.get('tp', 0)),
        str(micro.get('fp', 0)),
        str(micro.get('fn', 0)),
        ""
    ]
    if model_name:
        micro_row.insert(0, model_name)
    rows.append(micro_row)

    # Macro average
    macro = metrics.get("macro", {})
    macro_row = [
        "MACRO_AVG",
        f"{macro.get('precision', 0):.4f}",
        f"{macro.get('recall', 0):.4f}",
        f"{macro.get('f1', 0):.4f}",
        "", "", "", ""
    ]
    if model_name:
        macro_row.insert(0, model_name)
    rows.append(macro_row)

    return rows


def metrics_to_csv_comparison(metrics_files: List[str]) -> List[List[str]]:
    """Convert multiple metrics files to comparison CSV."""
    all_metrics = {}
    all_entity_types = set()

    # Load all metrics
    for file_path in metrics_files:
        model_name = Path(file_path).stem
        metrics = load_metrics(file_path)
        all_metrics[model_name] = metrics
        all_entity_types.update(metrics.get("per_class", {}).keys())

    model_names = list(all_metrics.keys())
    rows = []

    # Header
    header = ["entity_type"]
    for model in model_names:
        header.extend([f"{model}_P", f"{model}_R", f"{model}_F1"])
    rows.append(header)

    # Per-class metrics
    for entity_type in sorted(all_entity_types):
        row = [entity_type]
        for model in model_names:
            per_class = all_metrics[model].get("per_class", {})
            data = per_class.get(entity_type, {})
            row.extend([
                f"{data.get('precision', 0):.4f}",
                f"{data.get('recall', 0):.4f}",
                f"{data.get('f1', 0):.4f}"
            ])
        rows.append(row)

    # Micro average
    micro_row = ["MICRO_AVG"]
    for model in model_names:
        micro = all_metrics[model].get("micro", {})
        micro_row.extend([
            f"{micro.get('precision', 0):.4f}",
            f"{micro.get('recall', 0):.4f}",
            f"{micro.get('f1', 0):.4f}"
        ])
    rows.append(micro_row)

    # Macro average
    macro_row = ["MACRO_AVG"]
    for model in model_names:
        macro = all_metrics[model].get("macro", {})
        macro_row.extend([
            f"{macro.get('precision', 0):.4f}",
            f"{macro.get('recall', 0):.4f}",
            f"{macro.get('f1', 0):.4f}"
        ])
    rows.append(macro_row)

    return rows


def write_csv(rows: List[List[str]], output_file: str = None):
    """Write rows to CSV file or stdout."""
    if output_file:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"CSV written to: {output_file}")
    else:
        writer = csv.writer(sys.stdout)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NER metrics JSON to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file to stdout
  python metrics_to_csv.py metrics.json

  # Single file to CSV
  python metrics_to_csv.py metrics.json -o output.csv

  # Compare multiple models
  python metrics_to_csv.py gliner_metrics.json binder_metrics.json -o comparison.csv

  # Compare with custom output
  python metrics_to_csv.py metrics_finetuned.json metrics_test.json -o compare.csv
        """
    )
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input metrics JSON file(s)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file (default: stdout)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Force comparison mode even with single file'
    )

    args = parser.parse_args()

    if len(args.input_files) == 1 and not args.compare:
        # Single file mode
        metrics = load_metrics(args.input_files[0])
        model_name = Path(args.input_files[0]).stem
        rows = metrics_to_csv_single(metrics, model_name=None)
    else:
        # Comparison mode
        rows = metrics_to_csv_comparison(args.input_files)

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
