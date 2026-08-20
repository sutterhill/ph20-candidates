#!/usr/bin/env python3
"""Run one local Ridgey 600M checkpoint on the folded PH20 parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ridgey.serving import RidgeyPredictor, prepare_structure


ROOT = Path("/home/ubuntu/codex_ph20_20260820")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structure_path = ROOT / "work/esmfold2fast_parent/cif/PH20_soluble_36_483_construct.cif"
    source = prepare_structure(
        content=structure_path.read_text(),
        filename=structure_path.name,
        chain_id="A",
        name="PH20_parent",
    )
    predictor = RidgeyPredictor(
        checkpoint_path=args.checkpoint,
        artifact_dir=ROOT / "models/artifacts",
        device=args.device,
        model_name=args.model_name,
    )
    result = predictor.predict([source], token_budget=12_000)[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": args.model_name, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
