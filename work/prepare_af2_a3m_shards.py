#!/usr/bin/env python3
"""Prepare controlled candidate-specific A3Ms for local AF2-pTM folding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name: str | None = None
    pieces: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(pieces)
            name = line[1:].split()[0]
            pieces = []
        else:
            pieces.append(line.strip())
    if name is not None:
        records[name] = "".join(pieces)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-fasta", type=Path, required=True)
    parser.add_argument("--candidates-fasta", type=Path, required=True)
    parser.add_argument("--base-a3m", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    parent_records = read_fasta(args.parent_fasta)
    if len(parent_records) != 1:
        raise ValueError("parent FASTA must contain exactly one record")
    parent = next(iter(parent_records.values()))
    candidates = read_fasta(args.candidates_fasta)
    records = {"WT": parent, **candidates}
    if len(records) != 49:
        raise ValueError(f"expected WT plus 48 candidates, found {len(records)}")
    if len(set(records.values())) != len(records):
        raise ValueError("sequences are not unique")
    for name, sequence in candidates.items():
        if len(sequence) != len(parent):
            raise ValueError(f"{name}: length {len(sequence)} != parent {len(parent)}")
        if sum(left != right for left, right in zip(sequence, parent)) != 10:
            raise ValueError(f"{name}: not an exact ten-mutant")

    base_lines = args.base_a3m.read_text().splitlines()
    if not base_lines or not base_lines[0].startswith(">"):
        raise ValueError("invalid base A3M")
    next_header = next(
        (index for index, line in enumerate(base_lines[1:], 1) if line.startswith(">")),
        None,
    )
    if next_header is None:
        raise ValueError("base A3M contains no homologs")
    query = "".join(base_lines[1:next_header])
    query_ungapped = "".join(char for char in query if char != "-" and not char.islower())
    if query_ungapped != parent:
        raise ValueError("base A3M query does not match parent")

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.out_dir}")
    for shard in range(args.num_shards):
        (args.out_dir / "input_shards" / str(shard)).mkdir(parents=True)
        (args.out_dir / "result_shards" / str(shard)).mkdir(parents=True)
        (args.out_dir / "logs").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for ordinal, (name, sequence) in enumerate(records.items()):
        shard = ordinal % args.num_shards
        destination = args.out_dir / "input_shards" / str(shard) / f"{name}.a3m"
        destination.write_text(
            "\n".join([f">{name}", sequence, *base_lines[next_header:]]) + "\n"
        )
        manifest.append({
            "name": name,
            "shard": shard,
            "length": len(sequence),
            "mutations": sum(left != right for left, right in zip(sequence, parent)),
            "input_a3m": str(destination),
        })
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "records": len(manifest),
        "parent_length": len(parent),
        "base_msa_sequences": sum(line.startswith(">") for line in base_lines),
        "shard_counts": {
            str(shard): sum(row["shard"] == shard for row in manifest)
            for shard in range(args.num_shards)
        },
    }, indent=2))


if __name__ == "__main__":
    main()
