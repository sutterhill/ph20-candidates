#!/usr/bin/env python3
"""Generate a broad PH20 pool, then select 480 exact ten-mutation designs."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
MODEL_NAMES = ("base", "ens1", "ens2", "ens3", "ens4")
SEED = 20260820
RAW_TARGET = 20_000
SCORE_POOL_SIZE = 480
ATTEMPTS = 2_000_000
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}
HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
    "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}


def read_sequence() -> str:
    return "".join(
        line.strip() for line in (ROOT / "ph20.fasta").read_text().splitlines()
        if not line.startswith(">")
    )


def load_ensemble() -> tuple[list[dict], dict[str, dict]]:
    directory = ROOT / "work/ridgey_local_ensemble_broad_singles"
    models = [json.loads((directory / f"{name}.json").read_text()) for name in MODEL_NAMES]
    wild_types = [model["records"][0] for model in models]
    maps = [{record["name"]: record for record in model["records"]} for model in models]
    names = [record["name"] for record in models[0]["records"][1:]]
    scored: dict[str, dict] = {}
    for name in names:
        records = [mapping[name] for mapping in maps]
        scored[name] = {
            "stability": [record["stability"] - wt["stability"] for record, wt in zip(records, wild_types)],
            "ec": [record["ec_3.2.1.35"] - wt["ec_3.2.1.35"] for record, wt in zip(records, wild_types)],
            "solubility": [record["solubility"] - wt["solubility"] for record, wt in zip(records, wild_types)],
            "lm": [record["masked_lm"]["delta_mean_logp_vs_wt"] for record in records],
            "active_sites_intact": all(
                record["active_sites_1_indexed"] == wt["active_sites_1_indexed"]
                for record, wt in zip(records, wild_types)
            ),
        }
    return wild_types, scored


def ca_coordinates() -> np.ndarray:
    path = ROOT / "work/esmfold2fast_parent/cif/PH20_soluble_36_483_construct.cif"
    structure = gemmi.read_structure(str(path))
    chain = structure[0][0]
    coordinates = []
    for residue in chain:
        amino_acid = gemmi.find_tabulated_residue(residue.name).one_letter_code
        if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
            continue
        ca = residue.find_atom("CA", "*")
        coordinates.append([ca.pos.x, ca.pos.y, ca.pos.z])
    return np.asarray(coordinates, dtype=float)


def main() -> None:
    rng = random.Random(SEED)
    sequence = read_sequence()
    _, scored = load_ensemble()
    metadata = {
        row["mutation"]: row
        for row in csv.DictReader((ROOT / "work/ph20_broad_single_pool.csv").open())
    }
    coordinates = ca_coordinates()

    ev = np.load(ROOT / "work/ph20_ev.npz")
    ev_aa = str(ev["aa_order"])
    h, v = ev["h"], ev["V"]
    query_states = np.asarray([1 + ev_aa.index(amino_acid) for amino_acid in sequence], dtype=np.int64)
    baseline_v = v[np.arange(len(sequence)), :, query_states]
    baseline_total = baseline_v.sum(axis=0)

    single_candidates: list[dict] = []
    for mutation, values in scored.items():
        meta = metadata[mutation]
        mean_stability = statistics.mean(values["stability"])
        mean_ec = statistics.mean(values["ec"])
        mean_solubility = statistics.mean(values["solubility"])
        mean_lm = statistics.mean(values["lm"])
        if not values["active_sites_intact"]:
            continue
        if mean_stability < -0.01 or min(values["stability"]) < -0.06:
            continue
        if mean_ec < -0.0005 or min(values["ec"]) < -0.004:
            continue
        if mean_solubility < -0.0025:
            continue
        if mean_lm < -0.25 or sum(value >= 0 for value in values["lm"]) < 2:
            continue
        position = int(meta["position"])
        native, mutant = meta["native"], meta["mutant"]
        old_state = query_states[position - 1]
        new_state = 1 + ev_aa.index(mutant)
        old_v = v[position - 1, :, old_state]
        new_v = v[position - 1, :, new_state]
        mutation_score = (
            250.0 * mean_ec
            + 2.4 * mean_stability
            + 3.0 * mean_solubility
            + 0.10 * mean_lm
            + 0.02 * float(meta["potts_single_delta"])
            + 0.08 * math.log10(1.0 + 3801.0 * float(meta["msa_frequency"]))
        )
        single_candidates.append({
            "mutation": mutation,
            "position": position,
            "native": native,
            "mutant": mutant,
            "stability": values["stability"],
            "ec": values["ec"],
            "solubility": values["solubility"],
            "lm": values["lm"],
            "mean_stability": mean_stability,
            "mean_ec": mean_ec,
            "mean_solubility": mean_solubility,
            "mean_lm": mean_lm,
            "msa_frequency": float(meta["msa_frequency"]),
            "independent_delta": float(meta["independent_delta"]),
            "potts_single_delta": float(meta["potts_single_delta"]),
            "delta_v": new_v - old_v,
            "potts_field_delta": float(h[position - 1, new_state] - h[position - 1, old_state]),
            "potts_norm_delta": float(np.dot(new_v, new_v) - np.dot(old_v, old_v)),
            "charge_delta": CHARGE.get(mutant, 0) - CHARGE.get(native, 0),
            "surface_hydropathy_delta": (
                HYDROPATHY[mutant] - HYDROPATHY[native]
                if float(meta["relative_sasa"]) > 0.35 else 0.0
            ),
            "score": mutation_score,
        })

    by_position: dict[int, list[dict]] = defaultdict(list)
    for candidate in single_candidates:
        by_position[candidate["position"]].append(candidate)
    positions = sorted(by_position)
    if len(positions) < 15:
        raise RuntimeError(f"only {len(positions)} positions survived single-mutant consensus filtering")
    position_weights = [
        math.exp(min(3.0, max(-3.0, max(row["score"] for row in by_position[position]))))
        for position in positions
    ]

    designs: dict[tuple[str, ...], dict] = {}
    for _ in range(ATTEMPTS):
        remaining_positions = positions.copy()
        remaining_weights = position_weights.copy()
        chosen_positions: list[int] = []
        for _ in range(10):
            position = rng.choices(remaining_positions, weights=remaining_weights, k=1)[0]
            offset = remaining_positions.index(position)
            chosen_positions.append(position)
            remaining_positions.pop(offset)
            remaining_weights.pop(offset)
        chosen: list[dict] = []
        for position in chosen_positions:
            options = by_position[position]
            weights = [math.exp(min(3.0, max(-3.0, row["score"]))) for row in options]
            chosen.append(rng.choices(options, weights=weights, k=1)[0])

        # Spread changes across the construct and avoid directly contacting pairs.
        if sum(row["position"] <= 160 for row in chosen) < 2:
            continue
        if sum(160 < row["position"] <= 330 for row in chosen) < 2:
            continue
        if sum(row["position"] > 330 for row in chosen) < 2:
            continue
        if any(
            np.linalg.norm(coordinates[left["position"] - 1] - coordinates[right["position"] - 1]) < 6.0
            for index, left in enumerate(chosen)
            for right in chosen[index + 1 :]
        ):
            continue
        net_charge = sum(row["charge_delta"] for row in chosen)
        if abs(net_charge) > 3:
            continue
        surface_hydropathy = sum(row["surface_hydropathy_delta"] for row in chosen)
        if surface_hydropathy > 1.0:
            continue

        additive_stability = [sum(row["stability"][model] for row in chosen) for model in range(5)]
        additive_ec = [sum(row["ec"][model] for row in chosen) for model in range(5)]
        additive_solubility = [sum(row["solubility"][model] for row in chosen) for model in range(5)]
        additive_lm = [sum(row["lm"][model] for row in chosen) for model in range(5)]
        if min(additive_stability) < 0.0 or statistics.mean(additive_stability) < 0.20:
            continue
        if statistics.mean(additive_ec) < -0.002 or min(additive_ec) < -0.012:
            continue
        if statistics.mean(additive_solubility) < -0.004:
            continue
        if statistics.mean(additive_lm) < 0.0:
            continue

        total_delta_v = np.sum([row["delta_v"] for row in chosen], axis=0)
        potts_delta = (
            sum(row["potts_field_delta"] for row in chosen)
            + float(np.dot(baseline_total, total_delta_v))
            + 0.5 * float(np.dot(total_delta_v, total_delta_v))
            - 0.5 * sum(row["potts_norm_delta"] for row in chosen)
        )
        independent_delta = sum(row["independent_delta"] for row in chosen)
        if potts_delta < -60.0 or independent_delta < -22.0:
            continue
        mutations = tuple(sorted((row["mutation"] for row in chosen), key=lambda text: int(text[1:-1])))
        generation_score = (
            300.0 * statistics.mean(additive_ec)
            + 2.0 * min(additive_stability)
            + 0.8 * statistics.mean(additive_stability)
            + 3.0 * statistics.mean(additive_solubility)
            + 0.04 * statistics.mean(additive_lm)
            + 0.022 * potts_delta
            + 0.01 * independent_delta
            - 0.03 * abs(net_charge)
        )
        designs.setdefault(mutations, {
            "mutations": mutations,
            "chosen": chosen,
            "generation_score": generation_score,
            "additive_ec_mean": statistics.mean(additive_ec),
            "additive_ec_min": min(additive_ec),
            "additive_stability_mean": statistics.mean(additive_stability),
            "additive_stability_min": min(additive_stability),
            "additive_solubility_mean": statistics.mean(additive_solubility),
            "additive_lm_mean": statistics.mean(additive_lm),
            "potts_delta": potts_delta,
            "independent_delta": independent_delta,
            "net_charge_delta": net_charge,
            "surface_hydropathy_delta": surface_hydropathy,
        })
        if len(designs) >= RAW_TARGET:
            break

    if len(designs) < SCORE_POOL_SIZE:
        raise RuntimeError(f"generated only {len(designs)} designs; need at least {SCORE_POOL_SIZE}")
    ranked = sorted(designs.values(), key=lambda row: row["generation_score"], reverse=True)

    selected: list[dict] = []
    mutation_use: Counter[str] = Counter()
    for design in ranked:
        mutation_set = set(design["mutations"])
        if any(len(mutation_set & set(other["mutations"])) > 8 for other in selected):
            continue
        if any(mutation_use[mutation] >= int(0.80 * SCORE_POOL_SIZE) for mutation in mutation_set):
            continue
        selected.append(design)
        mutation_use.update(mutation_set)
        if len(selected) == SCORE_POOL_SIZE:
            break
    if len(selected) < SCORE_POOL_SIZE:
        raise RuntimeError(f"selected only {len(selected)} diverse designs from {len(designs)} generated")

    fields = [
        "name", "mutations", "generation_score", "additive_ec_mean", "additive_ec_min",
        "additive_stability_mean", "additive_stability_min", "additive_solubility_mean",
        "additive_lm_mean", "potts_delta", "independent_delta", "net_charge_delta",
        "surface_hydropathy_delta", "minimum_essential_ca_distance",
    ]
    raw_path = ROOT / "work/ph20_generated_10mut_pool.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields[:-1])
        writer.writeheader()
        for index, design in enumerate(ranked, 1):
            writer.writerow({
                "name": f"PH20_RAW_{index:05d}",
                **{key: (";".join(value) if key == "mutations" else value) for key, value in design.items() if key not in {"chosen"}},
            })

    csv_path = ROOT / "work/ph20_10mut_pool480.csv"
    fasta_path = ROOT / "work/ph20_10mut_pool480.fasta"
    essential = [112, 114, 177, 230, 250, 253]
    with csv_path.open("w", newline="") as csv_handle, fasta_path.open("w") as fasta_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fields)
        writer.writeheader()
        for index, design in enumerate(selected, 1):
            name = f"PH20_POOL_{index:04d}"
            designed = list(sequence)
            for row in design["chosen"]:
                designed[row["position"] - 1] = row["mutant"]
            minimum_distance = min(
                np.linalg.norm(coordinates[row["position"] - 1] - coordinates[site - 1])
                for row in design["chosen"] for site in essential
            )
            writer.writerow({
                "name": name,
                **{key: (";".join(value) if key == "mutations" else value) for key, value in design.items() if key not in {"chosen"}},
                "minimum_essential_ca_distance": minimum_distance,
            })
            fasta_handle.write(f">{name} mutations={'|'.join(design['mutations'])}\n{''.join(designed)}\n")

    print(json.dumps({
        "single_candidates": len(single_candidates),
        "mutable_positions": len(positions),
        "generated_unique_exact_ten_mutants": len(designs),
        "selected_for_five_model_scoring": len(selected),
        "raw_csv": str(raw_path),
        "pool_csv": str(csv_path),
        "pool_fasta": str(fasta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
