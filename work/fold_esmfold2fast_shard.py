#!/usr/bin/env python3
"""Fold one deterministic shard of an NGLY1 FASTA with local ESMFold2-Fast."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


MODEL_NAME = "biohub/ESMFold2-Fast"


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    pieces: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(pieces)))
            name = line[1:].split()[0]
            pieces = []
        else:
            pieces.append(line.strip())
    if name is not None:
        records.append((name, "".join(pieces)))
    return records


def scalar(value) -> float | None:
    if value is None:
        return None
    array = value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    finite = array.reshape(-1)[np.isfinite(array.reshape(-1))]
    if not finite.size:
        return None
    answer = float(finite.mean())
    return answer / 100.0 if answer > 1.0 else answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    records = read_fasta(Path(args.fasta))
    assigned = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]
    out_dir = Path(args.out_dir)
    cif_dir = out_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"shard_{args.shard_index:02d}.json"

    print(f"shard {args.shard_index}: loading {MODEL_NAME} for {len(assigned)} sequences", flush=True)
    model = ESMFold2Model.from_pretrained(MODEL_NAME).to(torch.device("cuda")).eval()
    builder = ESMFold2InputBuilder()
    summaries: list[dict] = []
    for ordinal, (name, sequence) in enumerate(assigned, 1):
        started = time.time()
        spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
        with torch.no_grad():
            result = builder.fold(
                model,
                spi,
                num_loops=args.num_loops,
                num_sampling_steps=args.num_sampling_steps,
                num_diffusion_samples=1,
                seed=args.seed,
            )
        cif_text = result.complex.to_mmcif()
        cif_path = cif_dir / f"{name}.cif"
        cif_path.write_text(cif_text)
        item = {
            "name": name,
            "length": len(sequence),
            "plddt": scalar(getattr(result, "plddt", None)),
            "ptm": scalar(getattr(result, "ptm", None)),
            "fold_seconds": time.time() - started,
            "num_loops": args.num_loops,
            "num_sampling_steps": args.num_sampling_steps,
            "seed": args.seed,
            "cif": str(cif_path),
        }
        summaries.append(item)
        summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
        print(
            f"shard {args.shard_index}: {ordinal}/{len(assigned)} {name} "
            f"pLDDT={item['plddt']:.4f} pTM={item['ptm']:.4f} seconds={item['fold_seconds']:.1f}",
            flush=True,
        )
        del result
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
