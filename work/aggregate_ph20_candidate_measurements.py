#!/usr/bin/env python3
"""Merge Ridgey, ESMFold2-Fast, MSA, and Potts measurements for PlayGod."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
OUT = ROOT / "outputs/ph20_48_10mut_designs"
MODEL_NAMES = ("base", "ens1", "ens2", "ens3", "ens4")


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


def potts_score(states: np.ndarray, h: np.ndarray, v: np.ndarray) -> float:
    selected_v = v[np.arange(len(states)), :, states]
    field = h[np.arange(len(states)), states].sum()
    total = selected_v.sum(axis=0)
    pair = 0.5 * ((total * total).sum() - (selected_v * selected_v).sum())
    return float(field + pair)


def pct_change(value: float, parent: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(parent) or abs(parent) < 1e-12:
        return None
    return 100.0 * (value - parent) / abs(parent)


def geometric(values: list[float]) -> float:
    finite = [max(value, 1e-12) for value in values if math.isfinite(value)]
    return math.exp(statistics.mean(math.log(value) for value in finite))


def severity_max(left: str, right: str) -> str:
    order = {"pass": 0, "watch": 1, "warning": 2}
    return left if order[left] >= order[right] else right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wt = "".join(line.strip() for line in (ROOT / "ph20.fasta").read_text().splitlines() if not line.startswith(">"))
    sequences = read_fasta(OUT / "ph20_48_10mut_designs.fasta")
    final_rows = list(csv.DictReader((OUT / "ph20_48_10mut_designs.csv").open()))
    final_by_design = {row["design"]: row for row in final_rows}
    plate_by_design = final_by_design

    likelihood_models = [json.loads((ROOT / "work/ridgey_structure_likelihood" / f"{name}.json").read_text()) for name in MODEL_NAMES]
    likelihood_maps = [{row["design"]: row for row in model["results"]} for model in likelihood_models]
    ridgey_models = [json.loads((ROOT / "work/ridgey_local_ensemble_pool480" / f"{name}.json").read_text()) for name in MODEL_NAMES]
    ridgey_maps = [{row["name"]: row for row in model["records"]} for model in ridgey_models]
    ridgey_wt = [model["records"][0] for model in ridgey_models]

    ev = np.load(ROOT / "work/ph20_ev.npz")
    aa_order = str(ev["aa_order"])
    h, v = ev["h"], ev["V"]
    wt_states = np.asarray([1 + aa_order.index(amino_acid) for amino_acid in wt], dtype=np.int64)
    wt_potts = potts_score(wt_states, h, v)
    manifest = json.loads((OUT / "ph20_48_manifest.json").read_text())
    parent_plddt = float(manifest["esmfold2fast_wt"]["plddt"])
    parent_ptm = float(manifest["esmfold2fast_wt"]["ptm"])

    mutation_catalog: dict[str, dict] = {}
    candidate_rows: list[dict] = []
    flat_rows: list[dict] = []
    for design in sorted(sequences, key=lambda name: int(name.rsplit("_", 1)[1])):
        sequence = sequences[design]
        final = final_by_design[design]
        plate = plate_by_design[design]
        source_name = final["source_name"]
        mutations = final["mutations"].split(";")
        positions = [int(mutation[1:-1]) for mutation in mutations]
        design_states = wt_states.copy()
        for mutation in mutations:
            design_states[int(mutation[1:-1]) - 1] = 1 + aa_order.index(mutation[-1])
        design_potts = potts_score(design_states, h, v)
        exact_potts_delta = design_potts - wt_potts

        likelihood_rows = [mapping[design] for mapping in likelihood_maps]
        per_likelihood_maps = [
            {row["mutation"]: row for row in result["per_residue"]}
            for result in likelihood_rows
        ]
        parent_mean_logp = statistics.mean(result["parent_structure"]["mean_log_probability"] for result in likelihood_rows)
        own_mean_logp = statistics.mean(result["own_structure"]["mean_log_probability"] for result in likelihood_rows)
        wt_mean_logp = statistics.mean(result["wt_matched_positions_on_parent"]["mean_log_probability"] for result in likelihood_rows)
        parent_geom = math.exp(parent_mean_logp)
        own_geom = math.exp(own_mean_logp)
        wt_geom = math.exp(wt_mean_logp)

        per_mutation: list[dict] = []
        structure_flags: list[dict] = []
        msa_flags: list[dict] = []
        single_potts_sum = 0.0
        independent_sum = 0.0
        for mutation in mutations:
            position = int(mutation[1:-1])
            index = position - 1
            native, mutant = mutation[0], mutation[-1]
            aa_index = aa_order.index(mutant)
            native_index = aa_order.index(native)
            site_rows = [mapping[mutation] for mapping in per_likelihood_maps]
            parent_site_logp = statistics.mean(row["candidate_on_parent_log_probability"] for row in site_rows)
            own_site_logp = statistics.mean(row["candidate_on_own_log_probability"] for row in site_rows)
            wt_site_logp = statistics.mean(row["wt_on_parent_log_probability"] for row in site_rows)
            parent_site_probability = math.exp(parent_site_logp)
            own_site_probability = math.exp(own_site_logp)
            wt_site_probability = math.exp(wt_site_logp)
            own_vs_wt_ratio = math.exp(own_site_logp - wt_site_logp)
            own_vs_parent_ratio = math.exp(own_site_logp - parent_site_logp)
            parent_vs_wt_ratio = math.exp(parent_site_logp - wt_site_logp)
            own_better_model_votes = sum(row["candidate_on_own_log_probability"] >= row["candidate_on_parent_log_probability"] for row in site_rows)
            wt_better_model_votes = sum(row["candidate_on_own_log_probability"] >= row["wt_on_parent_log_probability"] for row in site_rows)

            mutant_frequency = float(ev["frequencies"][index, aa_index])
            wt_frequency = float(ev["frequencies"][index, native_index])
            frequency_ratio = mutant_frequency / max(wt_frequency, 1e-12)
            independent_delta = float(ev["dE_independent"][index, aa_index])
            single_potts_delta = float(ev["dE_potts"][index, aa_index])
            reverted_states = design_states.copy()
            reverted_states[index] = wt_states[index]
            context_potts_delta = design_potts - potts_score(reverted_states, h, v)
            single_potts_sum += single_potts_delta
            independent_sum += independent_delta

            structure_level = "pass"
            structure_reasons: list[str] = []
            if own_site_probability < 0.005:
                structure_level = "warning"
                structure_reasons.append("own-backbone masked probability <0.5%")
            elif own_site_probability < 0.02:
                structure_level = severity_max(structure_level, "watch")
                structure_reasons.append("own-backbone masked probability <2%")
            if own_vs_parent_ratio < 0.50:
                structure_level = "warning"
                structure_reasons.append("own backbone fits >2x worse than parent backbone")
            elif own_vs_parent_ratio < 0.75:
                structure_level = severity_max(structure_level, "watch")
                structure_reasons.append("own backbone fits >25% worse than parent backbone")
            if own_vs_wt_ratio < 0.50:
                structure_level = "warning"
                structure_reasons.append("mutant fits own backbone >2x worse than WT residue fits WT backbone")
            elif own_vs_wt_ratio < 0.75:
                structure_level = severity_max(structure_level, "watch")
                structure_reasons.append("mutant fits own backbone >25% worse than WT reference")

            msa_level = "pass"
            msa_reasons: list[str] = []
            if mutant_frequency < 0.005:
                msa_level = "warning"
                msa_reasons.append("MSA frequency <0.5%")
            elif mutant_frequency < 0.02:
                msa_level = severity_max(msa_level, "watch")
                msa_reasons.append("MSA frequency <2%")
            if independent_delta < -4.0:
                msa_level = "warning"
                msa_reasons.append("independent-site log-odds < -4")
            elif independent_delta < -3.0:
                msa_level = severity_max(msa_level, "watch")
                msa_reasons.append("independent-site log-odds < -3")
            if single_potts_delta < -10.0:
                msa_level = "warning"
                msa_reasons.append("single-mutant Potts delta < -10")
            elif single_potts_delta < -8.0:
                msa_level = severity_max(msa_level, "watch")
                msa_reasons.append("single-mutant Potts delta < -8")
            if context_potts_delta < -10.0:
                msa_level = "warning"
                msa_reasons.append("design-context Potts contribution < -10")
            elif context_potts_delta < -8.0:
                msa_level = severity_max(msa_level, "watch")
                msa_reasons.append("design-context Potts contribution < -8")
            if float(ev["entropy_bits"][index]) < 1.0 and mutant_frequency < 0.05:
                msa_level = "warning"
                msa_reasons.append("rare substitution at a conserved position")

            combined_level = severity_max(structure_level, msa_level)
            diagnostic = {
                "mutation": mutation,
                "position": position,
                "native": native,
                "mutant": mutant,
                "structure_fit_level": structure_level,
                "structure_fit_reasons": structure_reasons,
                "msa_level": msa_level,
                "msa_reasons": msa_reasons,
                "combined_level": combined_level,
                "p_mutant_given_parent_structure": parent_site_probability,
                "p_mutant_given_own_structure": own_site_probability,
                "p_wt_given_parent_structure": wt_site_probability,
                "own_vs_parent_probability_ratio": own_vs_parent_ratio,
                "own_vs_wt_probability_ratio": own_vs_wt_ratio,
                "parent_vs_wt_probability_ratio": parent_vs_wt_ratio,
                "own_better_than_parent_model_votes": own_better_model_votes,
                "own_at_least_wt_model_votes": wt_better_model_votes,
                "msa_mutant_frequency": mutant_frequency,
                "msa_wt_frequency": wt_frequency,
                "msa_frequency_ratio_vs_wt": frequency_ratio,
                "msa_entropy_bits": float(ev["entropy_bits"][index]),
                "msa_gap_fraction": float(ev["gap_fraction"][index]),
                "independent_site_delta": independent_delta,
                "single_mutant_potts_delta": single_potts_delta,
                "design_context_potts_contribution": context_potts_delta,
            }
            per_mutation.append(diagnostic)
            if structure_level != "pass":
                structure_flags.append(diagnostic)
            if msa_level != "pass":
                msa_flags.append(diagnostic)
            catalog = mutation_catalog.setdefault(mutation, {
                "mutation": mutation,
                "position": position,
                "msa_mutant_frequency": mutant_frequency,
                "msa_wt_frequency": wt_frequency,
                "independent_site_delta": independent_delta,
                "single_mutant_potts_delta": single_potts_delta,
                "designs": [],
                "structure_warning_count": 0,
                "structure_watch_count": 0,
                "msa_warning_count": 0,
                "msa_watch_count": 0,
            })
            catalog["designs"].append(design)
            if structure_level == "warning":
                catalog["structure_warning_count"] += 1
            elif structure_level == "watch":
                catalog["structure_watch_count"] += 1
            if msa_level == "warning":
                catalog["msa_warning_count"] += 1
            elif msa_level == "watch":
                catalog["msa_watch_count"] += 1

        ridgey_records = [mapping[source_name] for mapping in ridgey_maps]
        stability = statistics.mean(record["stability"] for record in ridgey_records)
        parent_stability = statistics.mean(record["stability"] for record in ridgey_wt)
        solubility = statistics.mean(record["solubility"] for record in ridgey_records)
        parent_solubility = statistics.mean(record["solubility"] for record in ridgey_wt)
        ec_probability = statistics.mean(record["ec_3.2.1.35"] for record in ridgey_records)
        parent_ec_probability = statistics.mean(record["ec_3.2.1.35"] for record in ridgey_wt)
        enzyme_probability = statistics.mean(record["is_enzyme"] for record in ridgey_records)
        parent_enzyme_probability = statistics.mean(record["is_enzyme"] for record in ridgey_wt)
        plddt = float(final["esmfold2fast_plddt"])
        ptm = float(final["esmfold2fast_ptm"])
        mutation_frequencies = [row["msa_mutant_frequency"] for row in per_mutation]
        wt_frequencies = [row["msa_wt_frequency"] for row in per_mutation]
        overall_level = "pass"
        for row in per_mutation:
            overall_level = severity_max(overall_level, row["combined_level"])

        measurements = {
            "stability": {"value": stability, "parent": parent_stability, "delta": stability - parent_stability, "percent_change": pct_change(stability, parent_stability), "unit": "kcal/mol-equivalent", "good_direction": "higher"},
            "solubility": {"value": solubility, "parent": parent_solubility, "delta": solubility - parent_solubility, "percent_change": pct_change(solubility, parent_solubility), "unit": "probability", "good_direction": "higher"},
            "ec_3_2_1_35": {"value": ec_probability, "parent": parent_ec_probability, "delta": ec_probability - parent_ec_probability, "percent_change": pct_change(ec_probability, parent_ec_probability), "unit": "probability", "good_direction": "higher"},
            "is_enzyme": {"value": enzyme_probability, "parent": parent_enzyme_probability, "delta": enzyme_probability - parent_enzyme_probability, "percent_change": pct_change(enzyme_probability, parent_enzyme_probability), "unit": "probability", "good_direction": "higher"},
            "p_sequence_parent_structure": {"value": parent_geom, "parent": wt_geom, "delta": parent_geom - wt_geom, "percent_change": pct_change(parent_geom, wt_geom), "unit": "geometric mean probability per mutated residue", "good_direction": "higher", "mean_log_probability": parent_mean_logp, "mutation_site_pseudolikelihood": math.exp(max(-745.0, parent_mean_logp * 10))},
            "p_sequence_own_structure": {"value": own_geom, "parent": wt_geom, "delta": own_geom - wt_geom, "percent_change": pct_change(own_geom, wt_geom), "unit": "geometric mean probability per mutated residue", "good_direction": "higher", "mean_log_probability": own_mean_logp, "mutation_site_pseudolikelihood": math.exp(max(-745.0, own_mean_logp * 10)), "percent_change_vs_parent_backbone": pct_change(own_geom, parent_geom)},
            "sequence_only_masked_ppl": {"value": float(final["ridgey_masked_ppl_ratio_mean"]), "parent": 1.0, "delta": float(final["ridgey_masked_ppl_ratio_mean"]) - 1.0, "percent_change": 100.0 * (float(final["ridgey_masked_ppl_ratio_mean"]) - 1.0), "unit": "ratio vs WT at matched sites", "good_direction": "lower"},
            "esmfold2fast_plddt": {"value": plddt, "parent": parent_plddt, "delta": plddt - parent_plddt, "percent_change": pct_change(plddt, parent_plddt), "unit": "0-1", "good_direction": "higher"},
            "esmfold2fast_ptm": {"value": ptm, "parent": parent_ptm, "delta": ptm - parent_ptm, "percent_change": pct_change(ptm, parent_ptm), "unit": "0-1", "good_direction": "higher"},
            "active_pocket_ca_rmsd": {"value": float(final["active_pocket_ca_rmsd_angstrom"]), "parent": 0.0, "delta": float(final["active_pocket_ca_rmsd_angstrom"]), "percent_change": None, "unit": "angstrom", "good_direction": "lower"},
            "potts_statistical_score": {"value": design_potts, "parent": wt_potts, "delta": exact_potts_delta, "percent_change": None, "unit": "arbitrary statistical score", "good_direction": "higher"},
        }
        candidate = {
            "design": design,
            "well": plate["well"],
            "rank": int(design.rsplit("_", 1)[1]),
            "focus": "balanced stability / EC / solubility",
            "sequence": sequence,
            "mutations": mutations,
            "mutation_positions": positions,
            "mutation_count": 10,
            "identity_to_parent": (len(wt) - 10) / len(wt),
            "overall_flag_level": overall_level,
            "structure_fit_flag_count": len(structure_flags),
            "structure_warning_count": sum(row["structure_fit_level"] == "warning" for row in per_mutation),
            "structure_watch_count": sum(row["structure_fit_level"] == "watch" for row in per_mutation),
            "msa_flag_count": len(msa_flags),
            "msa_warning_count": sum(row["msa_level"] == "warning" for row in per_mutation),
            "msa_watch_count": sum(row["msa_level"] == "watch" for row in per_mutation),
            "warning_mutations": [row["mutation"] for row in per_mutation if row["combined_level"] == "warning"],
            "watch_mutations": [row["mutation"] for row in per_mutation if row["combined_level"] == "watch"],
            "measurements": measurements,
            "inverse_fold": {
                "definition": "one-at-a-time masked, structure-conditioned pseudolikelihood at the ten mutated sites; five Ridgey 600M checkpoints",
                "parent_structure_mean_log_probability": parent_mean_logp,
                "parent_structure_geometric_mean_probability": parent_geom,
                "parent_structure_mutation_site_pseudolikelihood": math.exp(max(-745.0, parent_mean_logp * 10)),
                "own_structure_mean_log_probability": own_mean_logp,
                "own_structure_geometric_mean_probability": own_geom,
                "own_structure_mutation_site_pseudolikelihood": math.exp(max(-745.0, own_mean_logp * 10)),
                "wt_matched_positions_mean_log_probability": wt_mean_logp,
                "wt_matched_positions_geometric_mean_probability": wt_geom,
                "own_vs_parent_percent_change": pct_change(own_geom, parent_geom),
                "own_vs_wt_percent_change": pct_change(own_geom, wt_geom),
                "parent_vs_wt_percent_change": pct_change(parent_geom, wt_geom),
                "flagged_residues": structure_flags,
            },
            "evolution": {
                "msa_sequences": 3801,
                "n_eff": float(ev["n_eff"]),
                "potts_rank": int(v.shape[1]),
                "l2_lambda": float(ev["best_lambda"]),
                "exact_potts_delta_vs_parent": exact_potts_delta,
                "sum_single_mutant_potts_delta": single_potts_sum,
                "epistatic_residual": exact_potts_delta - single_potts_sum,
                "independent_site_delta_sum": independent_sum,
                "minimum_mutant_frequency": min(mutation_frequencies),
                "geometric_mean_mutant_frequency": geometric(mutation_frequencies),
                "geometric_mean_wt_frequency": geometric(wt_frequencies),
                "frequency_percent_change_vs_wt": pct_change(geometric(mutation_frequencies), geometric(wt_frequencies)),
                "flagged_mutations": msa_flags,
            },
            "per_mutation": per_mutation,
            "structure_key": f"structures/{design}_esmfold2fast.cif",
        }
        candidate_rows.append(candidate)
        flat_rows.append({
            "well": candidate["well"],
            "design": design,
            "mutations": ";".join(mutations),
            "overall_flag_level": overall_level,
            "warning_mutations": ";".join(candidate["warning_mutations"]),
            "watch_mutations": ";".join(candidate["watch_mutations"]),
            "p_sequence_parent_structure": parent_geom,
            "p_sequence_parent_structure_percent_vs_parent": pct_change(parent_geom, wt_geom),
            "p_sequence_own_structure": own_geom,
            "p_sequence_own_structure_percent_vs_parent": pct_change(own_geom, wt_geom),
            "p_sequence_own_vs_parent_structure_percent": pct_change(own_geom, parent_geom),
            "msa_minimum_mutant_frequency": min(mutation_frequencies),
            "msa_geometric_mean_mutant_frequency": geometric(mutation_frequencies),
            "potts_delta_vs_parent": exact_potts_delta,
            "potts_epistatic_residual": exact_potts_delta - single_potts_sum,
            "stability": stability,
            "stability_percent_change": pct_change(stability, parent_stability),
            "ec_probability": ec_probability,
            "ec_percent_change": pct_change(ec_probability, parent_ec_probability),
            "solubility": solubility,
            "solubility_percent_change": pct_change(solubility, parent_solubility),
            "is_enzyme_probability": enzyme_probability,
            "is_enzyme_percent_change": pct_change(enzyme_probability, parent_enzyme_probability),
            "plddt": plddt,
            "plddt_percent_change": pct_change(plddt, parent_plddt),
            "ptm": ptm,
            "ptm_percent_change": pct_change(ptm, parent_ptm),
            "active_pocket_ca_rmsd_angstrom": float(final["active_pocket_ca_rmsd_angstrom"]),
            "sequence": sequence,
        })

    catalog_rows = []
    for mutation, row in sorted(mutation_catalog.items(), key=lambda item: int(item[0][1:-1])):
        row["usage_count"] = len(row.pop("designs"))
        catalog_rows.append(row)
    summary = {
        "candidate_count": len(candidate_rows),
        "all_exact_ten_mutants": all(row["mutation_count"] == 10 for row in candidate_rows),
        "structure_flagged_candidates": sum(row["structure_fit_flag_count"] > 0 for row in candidate_rows),
        "structure_warning_candidates": sum(row["structure_warning_count"] > 0 for row in candidate_rows),
        "msa_flagged_candidates": sum(row["msa_flag_count"] > 0 for row in candidate_rows),
        "msa_warning_candidates": sum(row["msa_warning_count"] > 0 for row in candidate_rows),
        "overall_warning_candidates": sum(row["overall_flag_level"] == "warning" for row in candidate_rows),
        "overall_watch_candidates": sum(row["overall_flag_level"] == "watch" for row in candidate_rows),
        "overall_pass_candidates": sum(row["overall_flag_level"] == "pass" for row in candidate_rows),
        "unique_mutations": len(catalog_rows),
        "warning_mutation_identities": [row["mutation"] for row in catalog_rows if row["structure_warning_count"] or row["msa_warning_count"]],
        "watch_mutation_identities": [row["mutation"] for row in catalog_rows if not (row["structure_warning_count"] or row["msa_warning_count"]) and (row["structure_watch_count"] or row["msa_watch_count"])],
    }
    payload = {
        "project": "Ph20 Candidates",
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "repository": "https://github.com/sutterhill/ph20-candidates",
            "analysis_commit": args.analysis_commit,
            "analysis_commit_url": f"https://github.com/sutterhill/ph20-candidates/commit/{args.analysis_commit}",
        },
        "methods": {
            "ridgey": "five local Ridgey v2 600M checkpoints",
            "inverse_fold": "one-at-a-time masked structure-conditioned pseudolikelihood at ten mutated positions",
            "folding": "ESMFold2-Fast, three loops and 50 sampling steps",
            "msa": "3,801 MMseqs sequences",
            "potts": "rank-32 low-rank Potts, L2 selected lambda=1.0, n_eff=1336.07",
            "color_rule": "green=favorable parent-relative direction; red=unfavorable; amber=mixed/watch; arbitrary Potts/log scores use deltas rather than misleading percentages",
        },
        "parent": {
            "name": "soluble human PH20 parent (P38567 residues 36-483)",
            "sequence": wt,
            "length": len(wt),
            "catalytic_triad": [112, 114, 230],
            "additional_essential_residues": [177, 250, 253],
            "protected_cysteines": [26, 190, 204, 317, 342, 347, 353, 401, 403, 409, 424, 430],
            "structure_key": "structures/WT_esmfold2fast.cif",
            "metrics": {
                "stability": statistics.mean(record["stability"] for record in ridgey_wt),
                "solubility": statistics.mean(record["solubility"] for record in ridgey_wt),
                "ec_3_2_1_35": statistics.mean(record["ec_3.2.1.35"] for record in ridgey_wt),
                "is_enzyme": statistics.mean(record["is_enzyme"] for record in ridgey_wt),
                "plddt": parent_plddt,
                "ptm": parent_ptm,
                "potts_score": wt_potts,
            },
        },
        "summary": summary,
        "mutation_catalog": catalog_rows,
        "candidates": sorted(candidate_rows, key=lambda row: (row["well"][0], int(row["well"][1:]))),
    }
    json_path = OUT / "ph20_candidate_measurements.json"
    csv_path = OUT / "ph20_candidate_measurements.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(flat_rows, key=lambda row: (row["well"][0], int(row["well"][1:]))))
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
