"""Versioned, budget-solvable FrozenLake maps with geometry-disjoint splits.

Generation never uses a model or its success scores. Sizes have balanced quotas;
p is uniform in [.6, .85] for each row. Rejections and resulting distributions
are reported, not hidden. Small maps necessarily repeat within train; all three
evaluation splits are internally unique. Held-out geometry is reserved first.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor/rllm_source"))
from rllm.environments.frozenlake.frozenlake import (
    generate_random_map, normalize_desc, shortest_path_length,
)


def geometry_hash(desc):
    rows = normalize_desc(desc)
    # Dimensions and every tile, including S/G, are part of the identity.
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def verify_task(task, max_steps=None):
    rows = normalize_desc(task["desc"])
    if len(rows) != task["size"] or any(len(row) != task["size"] for row in rows):
        raise ValueError("Task size does not match explicit map")
    if geometry_hash(rows) != task["geometry_hash"]:
        raise ValueError("Task geometry hash mismatch")
    distance = shortest_path_length(rows)
    budget = task["max_steps"] if max_steps is None else max_steps
    if distance is None or distance > budget:
        raise ValueError(f"Task shortest distance {distance} exceeds budget {budget}")
    if task["shortest_distance"] != distance:
        raise ValueError("Task shortest-distance metadata mismatch")
    return distance


def validate_dataset(data, max_steps=None):
    """Check contents and all cross-split geometry intersections before use."""
    owners, uids, report = {}, set(), {}
    for split, rows in data.items():
        local = set()
        for task in rows:
            verify_task(task, max_steps)
            if task["split"] != split or task["uid"] in uids:
                raise ValueError("Duplicate UID or inconsistent split label")
            uids.add(task["uid"])
            identity = task["geometry_hash"]
            if identity in owners and owners[identity] != split:
                raise ValueError("Map geometry occurs in multiple splits")
            if split != "train" and identity in local:
                raise ValueError("Duplicate geometry within evaluation split")
            owners[identity] = split
            local.add(identity)
        report[split] = {"rows": len(rows), "unique_geometries": len(local),
                         "duplicate_geometry_rows": len(rows) - len(local)}
    return report


def _split_rng(seed, split):
    value = int.from_bytes(hashlib.sha256(f"frozenlake-v2:{seed}:{split}".encode()).digest()[:8], "big")
    return np.random.default_rng(value)


def build_dataset(seed=20261014, train=10000, validation=100, test=100,
                  holdout=100, max_steps=10, max_attempts_per_row=100000):
    counts = dict(train=train, validation=validation, test=test, holdout=holdout)
    if any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError("Split sizes must be nonnegative integers")
    if max_steps < 1 or max_attempts_per_row < 1:
        raise ValueError("Budgets and attempt limits must be positive")
    # There are only 44 reachable 2x2 S/G geometries. At least one must remain
    # for train. Detect impossible unique-evaluation requests before sampling.
    eval_2x2 = sum((counts[s] + 7) // 8 for s in ("validation", "test", "holdout"))
    available_2x2 = 32 if max_steps == 1 else 44
    if eval_2x2 + int(train > 0) > available_2x2:
        raise ValueError("Unique evaluation 2x2 quota exhausts the finite geometry pool; reduce split counts")
    data, ownership, diagnostics = {}, {}, {}
    for split in ("holdout", "test", "validation", "train"):
        rng, n = _split_rng(seed, split), counts[split]
        sizes = np.array([2 + i % 8 for i in range(n)])
        rng.shuffle(sizes)
        rows, local = [], set()
        rejections = Counter(board_candidates=0, no_path=0, over_budget=0,
                             legacy_rejected=0, cross_split_geometry=0,
                             within_eval_geometry=0, map_draws=0)
        for index, size in enumerate(sizes):
            size, p = int(size), float(rng.uniform(.6, .85))
            for _ in range(max_attempts_per_row):
                map_seed = int(rng.integers(0, 2**31 - 1))
                desc, _, stats = generate_random_map(size, p, map_seed,
                    max_steps=max_steps, generation_mode="strict_bfs", return_stats=True)
                rejections.update(stats)
                rejections["map_draws"] += 1
                identity = geometry_hash(desc)
                if identity in ownership and ownership[identity] != split:
                    rejections["cross_split_geometry"] += 1
                    continue
                if split != "train" and identity in local:
                    rejections["within_eval_geometry"] += 1
                    continue
                break
            else:
                raise RuntimeError(f"Cannot fill {split} row {index}, size {size}; {dict(rejections)}")
            ownership[identity] = split
            local.add(identity)
            rows.append(dict(uid=f"v2_{split}_{index:05d}_{identity[:16]}", split=split,
                seed=map_seed, size=size, p=p, desc=desc, geometry_hash=identity,
                shortest_distance=shortest_path_length(desc), max_steps=max_steps,
                generation_mode="strict_bfs", dataset_version="frozenlake-v2"))
        data[split] = rows
        diagnostics[split] = dict(rejections)
    # Stable conventional ordering; RNG and reserve order above are explicit.
    data = {split: data[split] for split in counts}
    validation_report = validate_dataset(data, max_steps)
    manifest = dict(dataset_version="frozenlake-v2", seed=seed, max_steps=max_steps,
        sizes=counts, generation="BFS shortest distance <= max_steps",
        sizes_policy="Balanced size 2..9 quotas, randomized order within each split",
        p_policy="One p ~ Uniform(.6,.85) per row, held fixed during geometry rejection",
        reservation_order=["holdout", "test", "validation", "train"],
        geometry_identity="SHA256 of compact JSON desc; no rotational/reflection canonicalization",
        split_geometry_disjoint=True, evaluation_geometries_unique=True,
        training_duplicates_allowed=True, author_split_verified=False,
        holdout_policy="Reserved for a future final run; do not use for short-run selection",
        selection_policy="No model outputs or success scores used in dataset construction",
        comparison_note="New dataset distribution; v1-to-v2 scores are not paired comparisons",
        finite_small_map_note="Only 44 solvable 2x2 maps; reserving unique evaluation maps concentrates training duplicates",
        validation=validation_report, rejection_counts=diagnostics,
        distributions={split: dict(size_histogram=dict(sorted(Counter(t["size"] for t in rows).items())),
            shortest_distance_histogram=dict(sorted(Counter(t["shortest_distance"] for t in rows).items())),
            mean_shortest_distance=float(np.mean([t["shortest_distance"] for t in rows])) if rows else None,
            mean_p=float(np.mean([t["p"] for t in rows])) if rows else None)
            for split, rows in data.items()})
    return data, manifest


def save_dataset(path, data, manifest):
    path = Path(path)
    # Never silently overwrite the fixed v1 data or an existing v2 dataset.
    if path.name == "data.json" or path.exists() or path.with_name(path.stem + "_manifest.json").exists():
        raise FileExistsError(f"Refusing to replace a dataset: {path}")
    validate_dataset(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    manifest = {**manifest, "data_sha256": hashlib.sha256(encoded).hexdigest()}
    path.write_bytes(encoded)
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
