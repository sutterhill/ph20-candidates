#!/usr/bin/env python3
"""Score a PH20 mutation pool with one local Ridgey checkpoint.

The script is intentionally one-checkpoint-per-process so the five Ridgey 600M
replicas can run concurrently on separate H100s.  Besides the regular Ridgey
heads, it reports sequence-only masked-LM perplexity at the ten changed sites,
paired to WT masked at those same sites.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from ridgey.serving import PreparedProtein, RidgeyPredictor, prepare_structure


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
TARGET_EC = "3.2.1.35"


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


def changed_positions(wt: str, sequence: str) -> list[int]:
    if len(wt) != len(sequence):
        raise ValueError("sequence length mismatch")
    positions = [index for index, pair in enumerate(zip(wt, sequence)) if pair[0] != pair[1]]
    if not positions:
        raise ValueError("candidate is identical to WT")
    return positions


def probability(rows: list[dict], label: str) -> float:
    return next((float(row["probability"]) for row in rows if row["label"] == label), 0.0)


def compact(record: dict) -> dict:
    prediction = record["predictions"]
    return {
        "name": record["name"],
        "stability": float(prediction["stability"]),
        "solubility": float(prediction["solubility"]),
        "is_enzyme": float(prediction["is_enzyme"]),
        "ec_3.2.1.35": probability(prediction["ec_number"], TARGET_EC),
        "zinc": probability(prediction["cofactors"], "Zn(2+)"),
        "active_sites_1_indexed": [
            int(position) + 1 for position in prediction["active_site_positions_0_indexed"]
        ],
    }


@torch.inference_mode()
def score_masked_lm(
    predictor: RidgeyPredictor,
    base: PreparedProtein,
    wt: str,
    records: list[tuple[str, str]],
    *,
    pairs_per_batch: int = 6,
) -> dict[str, dict[str, float]]:
    """Simultaneously mask all ten changed sites, without structure input."""
    answer: dict[str, dict[str, float]] = {}
    device_type = predictor.device.type
    for start in range(0, len(records), pairs_per_batch):
        group = records[start : start + pairs_per_batch]
        proteins: list[PreparedProtein] = []
        row_positions: list[list[int]] = []
        row_names: list[tuple[str, str]] = []
        for name, sequence in group:
            positions = changed_positions(wt, sequence)
            proteins.append(replace(base, name=name, sequence=sequence, mmcif=""))
            proteins.append(replace(base, name=f"{name}__WT", sequence=wt, mmcif=""))
            row_positions.extend([positions, positions])
            row_names.extend([(name, "design"), (name, "wt")])

        batch = predictor._batch(proteins, eos=False)
        target_tokens = batch["sequence_tokens"].clone()
        masked_tokens = target_tokens.clone()
        for row, positions in enumerate(row_positions):
            offsets = torch.tensor([position + 1 for position in positions], device=predictor.device)
            masked_tokens[row, offsets] = int(predictor.tokenizer.mask_token_id)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device_type == "cuda"
        ):
            output = predictor.model.encoder(
                sequence_tokens=masked_tokens,
                sequence_id=batch["attention_mask"],
                structure=None,
                position_ids=batch["position_ids"],
                return_logits=True,
            )
        log_probabilities = output["logits"].float().log_softmax(dim=-1)
        for row, (name, kind) in enumerate(row_names):
            token_offsets = torch.tensor(
                [position + 1 for position in row_positions[row]], device=predictor.device
            )
            values = log_probabilities[row, token_offsets].gather(
                1, target_tokens[row, token_offsets].unsqueeze(1)
            ).squeeze(1)
            mean_logp = float(values.mean().cpu())
            answer.setdefault(name, {})[f"{kind}_mean_logp"] = mean_logp
            answer[name][f"{kind}_ppl"] = float(math.exp(min(50.0, -mean_logp)))
        del batch, target_tokens, masked_tokens, output, log_probabilities

    for values in answer.values():
        values["delta_mean_logp_vs_wt"] = values["design_mean_logp"] - values["wt_mean_logp"]
        values["ppl_ratio_vs_wt"] = values["design_ppl"] / values["wt_ppl"]
    return answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fasta",
        default=str(ROOT / "work/ph20_10mut_pool.fasta"),
        help="Candidate FASTA; every sequence is paired to WT for masked-LM scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wt = read_fasta(ROOT / "ph20.fasta")[0][1]
    designs = read_fasta(Path(args.fasta))
    source = prepare_structure(
        content=(ROOT / "work/esmfold2fast_parent/cif/PH20_soluble_36_483_construct.cif").read_text(),
        filename="PH20_soluble_36_483_construct.cif",
        chain_id="A",
        name="WT",
    )
    if source.sequence != wt:
        raise ValueError("WT FASTA and structure sequence differ")

    print(f"{args.model_name}: loading {args.checkpoint} on {args.device}", flush=True)
    predictor = RidgeyPredictor(
        checkpoint_path=args.checkpoint,
        artifact_dir=ROOT / "models/artifacts",
        device=args.device,
        model_name=args.model_name,
    )
    proteins = [source] + [replace(source, name=name, sequence=sequence, mmcif="") for name, sequence in designs]
    summaries: list[dict] = []
    for start in range(0, len(proteins), 16):
        chunk = proteins[start : start + 16]
        summaries.extend(compact(item) for item in predictor.predict(chunk, token_budget=12_000))
        print(f"{args.model_name}: annotation {min(start + len(chunk), len(proteins))}/{len(proteins)}", flush=True)

    lm = score_masked_lm(predictor, source, wt, designs)
    for item in summaries:
        if item["name"] in lm:
            item["masked_lm"] = lm[item["name"]]
    result = {
        "model": args.model_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "target_ec": TARGET_EC,
        "ppl_definition": "sequence-only simultaneous-mask perplexity at all changed sites; paired WT uses identical masked positions",
        "records": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.model_name}: wrote {output}", flush=True)


if __name__ == "__main__":
    main()
