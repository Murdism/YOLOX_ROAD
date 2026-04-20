#!/usr/bin/env python3
"""
Remap EMT fine-grained COCO annotations into a 3-class taxonomy.

The original EMT dataset uses 9 fine-grained categories. This script
consolidates them into 3 functional classes optimized for detection and
downstream tracking:

    VulnerableRoadUser : Pedestrian, Cyclist
    Two-Wheeler        : Motorbike, Small motorised vehicle
    Vehicle            : Car, Medium / Large vehicle, Bus, Emergency vehicle

The script also verifies the remapping by comparing per-class annotation
counts before and after — the sum of source category counts must equal
the resulting superclass count, otherwise annotations were lost.

Usage:
    python remap_emt_to_3class.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ANN_DIR = Path("datasets/EMT/annotations/detections_new")

INPUTS_TO_OUTPUTS = {
    ANN_DIR / "train.json": ANN_DIR / "train_3class.json",
    ANN_DIR / "test.json":  ANN_DIR / "test_3class.json",
}

# Mapping from fine-grained category name -> new superclass name.
NAME_TO_SUPERCLASS = {
    # Vulnerable road users
    "Pedestrian": "VulnerableRoadUser",
    "Cyclist":    "VulnerableRoadUser",
    # Two/three-wheeled motorized
    "Motorbike":               "Two-Wheeler",
    "Small_motorised_vehicle": "Two-Wheeler",
    # Four-wheel motorized
    "Car":               "Vehicle",
    "Medium_vehicle":    "Vehicle",
    "Large_vehicle":     "Vehicle",
    "Bus":               "Vehicle",
    "Emergency_vehicle": "Vehicle",
}

NEW_CATEGORIES = [
    {"id": 1, "name": "VulnerableRoadUser", "supercategory": "person"},
    {"id": 2, "name": "Two-Wheeler",        "supercategory": "vehicle"},
    {"id": 3, "name": "Vehicle",            "supercategory": "vehicle"},
]


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #

def count_by_category(data: dict) -> Counter[str]:
    """Return per-category-name annotation counts for a COCO payload."""
    id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    return Counter(id_to_name[ann["category_id"]] for ann in data["annotations"])


def remap_annotations(input_path: Path, output_path: Path) -> dict:
    """
    Remap a COCO annotation file into the 3-class taxonomy.

    Returns the in-memory remapped data so the caller can run additional
    verification without re-reading the output file.
    """
    print(f"\n[remap] {input_path.name} -> {output_path.name}")

    with input_path.open() as f:
        data = json.load(f)

    new_name_to_id = {c["name"]: c["id"] for c in NEW_CATEGORIES}

    # Sanity: every target superclass must exist in NEW_CATEGORIES
    missing_targets = set(NAME_TO_SUPERCLASS.values()) - set(new_name_to_id)
    if missing_targets:
        raise ValueError(
            f"Mapping target(s) not found in NEW_CATEGORIES: {missing_targets}. "
            f"Available: {sorted(new_name_to_id)}"
        )

    # Warn if dataset has source categories not covered by the mapping
    old_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    unmapped_sources = set(old_id_to_name.values()) - set(NAME_TO_SUPERCLASS)
    if unmapped_sources:
        print(f"  WARNING: unmapped source categories: {sorted(unmapped_sources)}")

    # Capture pre-remap counts (per fine-grained source name)
    before_counts = count_by_category(data)

    # Remap annotations
    remapped = 0
    skipped_breakdown: Counter[str] = Counter()
    for ann in data["annotations"]:
        old_name = old_id_to_name.get(ann["category_id"])
        new_name = NAME_TO_SUPERCLASS.get(old_name)
        if new_name is not None:
            ann["category_id"] = new_name_to_id[new_name]
            remapped += 1
        else:
            skipped_breakdown[old_name or "<unknown>"] += 1

    data["categories"] = NEW_CATEGORIES

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f)

    print(f"  remapped: {remapped:,}")
    print(f"  skipped:  {sum(skipped_breakdown.values()):,}")
    if skipped_breakdown:
        for name, count in skipped_breakdown.most_common():
            print(f"    {name}: {count:,}")

    # Verify by comparing pre-remap source counts vs post-remap superclass counts
    verify_remap_consistency(before_counts, data, source_name=input_path.name)

    return data


def verify_remap_consistency(
    before_counts: Counter[str],
    after_data: dict,
    source_name: str,
) -> None:
    """
    Compare source category sums to resulting superclass counts.

    For each new superclass, the sum of annotations from its source
    categories must equal the count of annotations now labelled with
    that superclass — otherwise the remap silently lost data.
    """
    print(f"\n[verify] before -> after for {source_name}")

    after_counts = count_by_category(after_data)

    # Group source categories by their target superclass
    superclass_to_sources: dict[str, list[str]] = {}
    for source, target in NAME_TO_SUPERCLASS.items():
        superclass_to_sources.setdefault(target, []).append(source)

    name_width = max(len(n) for n in NAME_TO_SUPERCLASS) + 2

    all_match = True
    for new_cat in NEW_CATEGORIES:
        super_name = new_cat["name"]
        sources = superclass_to_sources.get(super_name, [])

        expected = sum(before_counts.get(s, 0) for s in sources)
        actual = after_counts.get(super_name, 0)
        status = "OK" if expected == actual else "MISMATCH"
        if expected != actual:
            all_match = False

        print(f"\n  {super_name}  [{status}]")
        for source in sources:
            count = before_counts.get(source, 0)
            print(f"    {source:<{name_width}} : {count:>10,}")
        print(f"    {'-' * name_width}   {'-' * 10}")
        print(f"    {'expected sum':<{name_width}} : {expected:>10,}")
        print(f"    {'actual after':<{name_width}} : {actual:>10,}")

    print()
    if all_match:
        print("  All superclass counts match source sums — remap verified.")
    else:
        print("  WARNING: Counts do NOT match. Investigate before training.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    for input_path, output_path in INPUTS_TO_OUTPUTS.items():
        remap_annotations(input_path, output_path)


if __name__ == "__main__":
    main()