#!/usr/bin/env python3
"""Select 120 diverse consensus Ridgey winners for PH20 structure prediction."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
MODEL_NAMES = ("base", "ens1", "ens2", "ens3", "ens4")
N_FOLD = 120


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    name: str | None = None
    pieces: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                result[name] = "".join(pieces)
            name = line[1:].split()[0]
            pieces = []
        else:
            pieces.append(line.strip())
    if name is not None:
        result[name] = "".join(pieces)
    return result


def joined(values: list[float]) -> str:
    return ";".join(f"{value:.9g}" for value in values)


def main() -> None:
    directory = ROOT / "work/ridgey_local_ensemble_pool400"
    models = [json.loads((directory / f"{name}.json").read_text()) for name in MODEL_NAMES]
    wild_types = [model["records"][0] for model in models]
    record_maps = [{record["name"]: record for record in model["records"]} for model in models]
    sequences = read_fasta(ROOT / "work/ph20_10mut_pool400.fasta")
    metadata = {
        row["name"]: row
        for row in csv.DictReader((ROOT / "work/ph20_10mut_pool400.csv").open())
    }

    passing: list[dict] = []
    rejection_counts: Counter[str] = Counter()
    for name, sequence in sequences.items():
        records = [mapping[name] for mapping in record_maps]
        stability = [record["stability"] - wt["stability"] for record, wt in zip(records, wild_types)]
        ec_absolute = [record["ec_3.2.1.35"] for record in records]
        ec = [value - wt["ec_3.2.1.35"] for value, wt in zip(ec_absolute, wild_types)]
        solubility = [record["solubility"] - wt["solubility"] for record, wt in zip(records, wild_types)]
        lm = [record["masked_lm"]["delta_mean_logp_vs_wt"] for record in records]
        ppl_ratio = [record["masked_lm"]["ppl_ratio_vs_wt"] for record in records]
        enzyme = [record["is_enzyme"] for record in records]
        active_intact = all(
            record["active_sites_1_indexed"] == wt["active_sites_1_indexed"]
            for record, wt in zip(records, wild_types)
        )
        ec_up_models = sum(value >= 0.0 for value in ec)
        failed: list[str] = []
        if min(stability) <= 0.0: failed.append("stability_not_up_all_5")
        if statistics.mean(ec) < 0.0: failed.append("ec_mean_down")
        if min(ec) < -0.0018: failed.append("ec_worst_replica")
        if ec_up_models < 4: failed.append("ec_less_than_4_of_5")
        if statistics.mean(solubility) < 0.0: failed.append("solubility_mean_down")
        if statistics.mean(lm) < 0.0: failed.append("masked_lm_mean_down")
        if min(enzyme) < 0.995: failed.append("enzyme_probability")
        if not active_intact: failed.append("active_site_call_changed")
        if failed:
            rejection_counts.update(failed)
            continue
        meta = metadata[name]
        score = (
            400.0 * statistics.mean(ec)
            + 2.0 * min(stability)
            + 0.8 * statistics.mean(stability)
            + 6.0 * statistics.mean(solubility)
            + 0.06 * statistics.mean(lm)
            + 0.012 * float(meta["potts_delta"])
        )
        passing.append({
            "name": name,
            "sequence": sequence,
            "mutations": meta["mutations"],
            "selection_score": score,
            "ridgey_stability_delta_mean": statistics.mean(stability),
            "ridgey_stability_delta_min": min(stability),
            "ridgey_stability_delta_by_model": joined(stability),
            "ridgey_ec_probability_mean": statistics.mean(ec_absolute),
            "ridgey_ec_delta_mean": statistics.mean(ec),
            "ridgey_ec_delta_min": min(ec),
            "ridgey_ec_non_decreasing_models": ec_up_models,
            "ridgey_ec_delta_by_model": joined(ec),
            "ridgey_solubility_delta_mean": statistics.mean(solubility),
            "ridgey_solubility_delta_min": min(solubility),
            "ridgey_solubility_delta_by_model": joined(solubility),
            "ridgey_masked_lm_delta_mean": statistics.mean(lm),
            "ridgey_masked_ppl_ratio_mean": statistics.mean(ppl_ratio),
            "ridgey_enzyme_probability_min": min(enzyme),
            "active_sites_all_models": "114",
            "potts_delta": float(meta["potts_delta"]),
            "independent_delta": float(meta["independent_delta"]),
            "net_charge_delta": int(meta["net_charge_delta"]),
            "surface_hydropathy_delta": float(meta["surface_hydropathy_delta"]),
            "minimum_essential_ca_distance": float(meta["minimum_essential_ca_distance"]),
        })

    selected: list[dict] = []
    mutation_use: Counter[str] = Counter()
    for row in sorted(passing, key=lambda item: item["selection_score"], reverse=True):
        mutations = set(row["mutations"].split(";"))
        if any(len(mutations & set(other["mutations"].split(";"))) > 8 for other in selected):
            continue
        if any(mutation_use[mutation] >= 116 for mutation in mutations):
            continue
        selected.append(row)
        mutation_use.update(mutations)
        if len(selected) == N_FOLD:
            break
    if len(selected) < N_FOLD:
        raise RuntimeError(json.dumps({
            "error": f"selected only {len(selected)} fold candidates from {len(passing)} passing",
            "rejection_counts": rejection_counts,
        }, indent=2))

    csv_path = ROOT / "work/ph20_fold128_prefold.csv"
    fasta_path = ROOT / "work/ph20_fold128_prefold.fasta"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [key for key in selected[0] if key != "sequence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: value for key, value in row.items() if key != "sequence"})
    parent = read_fasta(ROOT / "ph20.fasta")["PH20_soluble_36_483_construct"]
    with fasta_path.open("w") as handle:
        handle.write(f">WT\n{parent}\n")
        for row in selected:
            handle.write(f">{row['name']} mutations={row['mutations'].replace(';', '|')}\n{row['sequence']}\n")
    print(json.dumps({
        "input_scored": len(sequences),
        "passing_all_hard_gates": len(passing),
        "selected_for_folding": len(selected),
        "rejection_counts": rejection_counts,
        "csv": str(csv_path),
        "fasta": str(fasta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
