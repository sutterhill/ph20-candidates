#!/usr/bin/env python3
"""Structure-conditioned masked pseudolikelihood for final PH20 designs.

For every candidate and each of its ten changed residues, score the candidate
amino acid with exactly that residue masked while conditioning on either the WT
ESMFold2-Fast backbone or the candidate's own ESMFold2-Fast backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from ridgey.serving import PreparedProtein, RidgeyPredictor, prepare_structure


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
OUT = ROOT / "outputs/ph20_48_10mut_designs"


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


def mutation_positions(wt: str, sequence: str) -> list[int]:
    positions = [index for index, (native, mutant) in enumerate(zip(wt, sequence)) if native != mutant]
    if len(positions) != 10:
        raise ValueError(f"expected ten mutations, got {len(positions)}")
    return positions


@torch.inference_mode()
def score_positions(
    predictor: RidgeyPredictor,
    protein: PreparedProtein,
    sequence: str,
    positions: list[int],
) -> list[dict]:
    proteins = [replace(protein, name=f"{protein.name}_{position + 1}") for position in positions]
    batch = predictor._batch(proteins, eos=False)
    target_tokens = batch["sequence_tokens"].clone()
    masked_tokens = target_tokens.clone()
    for row, position in enumerate(positions):
        masked_tokens[row, position + 1] = int(predictor.tokenizer.mask_token_id)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=predictor.device.type == "cuda"
    ):
        output = predictor.model.encoder(
            sequence_tokens=masked_tokens,
            sequence_id=batch["attention_mask"],
            structure=batch["structure"],
            position_ids=batch["position_ids"],
            return_logits=True,
        )
    log_probabilities = output["logits"].float().log_softmax(dim=-1)
    scores: list[dict] = []
    for row, position in enumerate(positions):
        token = target_tokens[row, position + 1]
        logp = float(log_probabilities[row, position + 1, token].cpu())
        scores.append({
            "position_1_indexed": position + 1,
            "amino_acid": sequence[position],
            "log_probability": logp,
            "probability": math.exp(logp),
        })
    return scores


def aggregate(scores: list[dict]) -> dict:
    logps = [row["log_probability"] for row in scores]
    total = sum(logps)
    mean = total / len(logps)
    return {
        "sum_log_probability": total,
        "mean_log_probability": mean,
        "geometric_mean_probability": math.exp(mean),
        "mutation_site_pseudolikelihood": math.exp(max(-745.0, total)),
        "perplexity": math.exp(min(50.0, -mean)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wt = "".join(line.strip() for line in (ROOT / "ph20.fasta").read_text().splitlines() if not line.startswith(">"))
    sequences = read_fasta(OUT / "ph20_48_10mut_designs.fasta")
    plate_rows = list(csv.DictReader((OUT / "ph20_48_10mut_designs.csv").open()))
    design_to_well = {row["design"]: row["well"] for row in plate_rows}

    parent = prepare_structure(
        content=(OUT / "structures/WT_esmfold2fast.cif").read_text(),
        filename="WT_esmfold2fast.cif",
        chain_id="A",
        name="WT",
    )
    if parent.sequence != wt:
        raise ValueError("parent structure and WT FASTA differ")
    print(f"{args.model_name}: loading checkpoint", flush=True)
    predictor = RidgeyPredictor(
        checkpoint_path=args.checkpoint,
        artifact_dir=ROOT / "models/artifacts",
        device=args.device,
        model_name=args.model_name,
    )

    unique_positions = sorted({position for sequence in sequences.values() for position in mutation_positions(wt, sequence)})
    wt_scores = score_positions(predictor, parent, wt, unique_positions)
    wt_by_position = {row["position_1_indexed"]: row for row in wt_scores}

    results: list[dict] = []
    for ordinal, (name, sequence) in enumerate(sequences.items(), 1):
        positions = mutation_positions(wt, sequence)
        own_path = OUT / "structures" / f"{name}_esmfold2fast.cif"
        own = prepare_structure(
            content=own_path.read_text(),
            filename=own_path.name,
            chain_id="A",
            name=name,
        )
        if own.sequence != sequence:
            raise ValueError(f"{name}: own structure and candidate FASTA differ")
        on_parent = replace(parent, name=f"{name}_on_parent", sequence=sequence, mmcif="")
        parent_scores = score_positions(predictor, on_parent, sequence, positions)
        own_scores = score_positions(predictor, own, sequence, positions)
        per_residue: list[dict] = []
        for parent_row, own_row in zip(parent_scores, own_scores):
            position = parent_row["position_1_indexed"]
            wt_row = wt_by_position[position]
            per_residue.append({
                "position_1_indexed": position,
                "mutation": f"{wt[position - 1]}{position}{sequence[position - 1]}",
                "wt_on_parent_probability": wt_row["probability"],
                "wt_on_parent_log_probability": wt_row["log_probability"],
                "candidate_on_parent_probability": parent_row["probability"],
                "candidate_on_parent_log_probability": parent_row["log_probability"],
                "candidate_on_own_probability": own_row["probability"],
                "candidate_on_own_log_probability": own_row["log_probability"],
            })
        results.append({
            "design": name,
            "well": design_to_well[name],
            "parent_structure": aggregate(parent_scores),
            "own_structure": aggregate(own_scores),
            "wt_matched_positions_on_parent": aggregate([wt_by_position[position + 1] for position in positions]),
            "per_residue": per_residue,
        })
        print(f"{args.model_name}: {ordinal}/{len(sequences)} {name}", flush=True)

    output = {
        "model": args.model_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "definition": "one-at-a-time masked structure-conditioned pseudolikelihood at the ten mutated residues",
        "parent_structure": "WT ESMFold2-Fast prediction",
        "own_structure": "candidate ESMFold2-Fast prediction",
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{args.model_name}: wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
