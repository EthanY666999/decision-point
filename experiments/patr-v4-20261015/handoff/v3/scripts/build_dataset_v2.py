#!/usr/bin/env python3
"""Build a separate v2 map manifest; never overwrite artifacts/data.json."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_v2 import ROOT, build_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/data_v2.json")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--holdout", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    data, manifest = build_dataset(seed=args.seed, train=args.train, validation=args.validation,
        test=args.test, holdout=args.holdout, max_steps=args.max_steps)
    manifest = save_dataset(args.output, data, manifest)
    print(json.dumps({"path": str(args.output), "sha256": manifest["data_sha256"],
                     "validation": manifest["validation"],
                     "distributions": manifest["distributions"],
                     "rejection_counts": manifest["rejection_counts"]}, indent=2))


if __name__ == "__main__":
    main()
