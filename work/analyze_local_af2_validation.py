#!/usr/bin/env python3
"""Analyze local AF2-pTM folds and merge them into a candidate payload."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import gemmi
import numpy as np


CONFIGS = {
    "ph20": {
        "output_subdir": "ph20_48_10mut_designs",
        "payload": "ph20_candidate_measurements.json",
        "fasta": "ph20_48_10mut_designs.fasta",
        "protected_json": "work/ph20_protected_positions.json",
        "repository": "https://github.com/sutterhill/ph20-candidates",
        "domains": {"gh56": (1, 335), "c_terminal": (336, None)},
        "exclude_high_confidence": [],
        "protected_key": "active_pocket_positions_1_indexed",
        "essential_keys": ["essential_positions_1_indexed"],
        "plddt_tolerance": 0.010,
        "ptm_tolerance": 0.020,
        "max_domain_rmsd": 1.50,
        "protected_rmsd": 0.75,
        "essential_rmsd": 0.75,
        "gate_protected_rmsd": True,
        "protected_label": "active pocket",
    },
    "ngly1": {
        "output_subdir": "ngly1_48_10mut_designs",
        "payload": "ngly1_candidate_measurements.json",
        "fasta": "ngly1_48_10mut_designs.fasta",
        "protected_json": "work/ngly1_protected_positions.json",
        "repository": "https://github.com/sutterhill/ngly1-candidates",
        "domains": {"nterm": (1, 111), "core": (169, 453), "paw": (454, None)},
        "exclude_high_confidence": [(112, 168)],
        "protected_key": "protected_positions_1_indexed",
        "essential_keys": ["catalytic_positions", "zinc_ligands"],
        "plddt_tolerance": 0.010,
        "ptm_tolerance": 0.030,
        "max_domain_rmsd": 2.50,
        "protected_rmsd": 1.50,
        "essential_rmsd": 0.75,
        "gate_protected_rmsd": False,
        "protected_label": "catalytic/Zn protection shell",
    },
}


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


def ca_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    structure = gemmi.read_structure(str(path))
    chain = structure[0][0]
    coordinates: list[list[float]] = []
    b_factors: list[float] = []
    for residue in chain:
        amino_acid = gemmi.find_tabulated_residue(residue.name).one_letter_code
        if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
            continue
        ca = residue.find_atom("CA", "*")
        coordinates.append([ca.pos.x, ca.pos.y, ca.pos.z])
        b_factors.append(float(ca.b_iso))
    return np.asarray(coordinates, dtype=float), np.asarray(b_factors, dtype=float)


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(reference).all(axis=1) & np.isfinite(mobile).all(axis=1)
    ref = reference[valid]
    mob = mobile[valid]
    if len(ref) < 3:
        raise ValueError("RMSD mask contains fewer than three residues")
    ref0 = ref - ref.mean(axis=0)
    mob0 = mob - mob.mean(axis=0)
    u, _, vt = np.linalg.svd(mob0.T @ ref0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return float(np.sqrt(np.mean(np.sum((mob0 @ rotation - ref0) ** 2, axis=1))))


def pct_change(value: float, parent: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(parent) or abs(parent) < 1e-12:
        return None
    return 100.0 * (value - parent) / abs(parent)


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def metric(value: float, parent: float, unit: str, good_direction: str) -> dict:
    return {
        "value": value,
        "parent": parent,
        "delta": value - parent,
        "percent_change": pct_change(value, parent),
        "unit": unit,
        "good_direction": good_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-commit", required=True)
    args = parser.parse_args()
    config = CONFIGS[args.project]
    root = args.root.resolve()
    af2_dir = root / "work/af2_local_v1"
    out = root / "outputs" / config["output_subdir"]
    payload_path = out / config["payload"]
    payload = json.loads(payload_path.read_text())
    candidates = read_fasta(out / config["fasta"])
    sequences = {"WT": payload["parent"]["sequence"], **candidates}
    if len(sequences) != 49:
        raise ValueError(f"expected 49 sequences, found {len(sequences)}")

    protected_data = json.loads((root / config["protected_json"]).read_text())
    protected_positions = set(protected_data[config["protected_key"]])
    essential_positions: set[int] = set()
    for key in config["essential_keys"]:
        values = protected_data[key]
        essential_positions.update(int(value) for value in (values.keys() if isinstance(values, dict) else values))

    result_root = af2_dir / "result_shards"
    results: dict[str, dict] = {}
    for name in sequences:
        scores_path = find_one(result_root, f"*/{name}_scores_rank_001_*.json")
        pdb_path = find_one(result_root, f"*/{name}_unrelaxed_rank_001_*.pdb")
        scores = json.loads(scores_path.read_text())
        plddt = np.asarray(scores["plddt"], dtype=float)
        if plddt.max() <= 1.0:
            plddt *= 100.0
        coordinates, pdb_plddt = ca_coordinates(pdb_path)
        if len(coordinates) != len(sequences[name]) or len(plddt) != len(coordinates):
            raise ValueError(f"{name}: inconsistent output length")
        if not np.isfinite(plddt).all() or not np.isfinite(coordinates).all():
            raise ValueError(f"{name}: non-finite AF2 output")
        results[name] = {
            "scores_path": scores_path,
            "pdb_path": pdb_path,
            "coordinates": coordinates,
            "plddt_array": plddt,
            "mean_plddt": float(plddt.mean() / 100.0),
            "ptm": float(scores["ptm"]),
        }

    wt = results["WT"]
    positions = np.arange(1, len(wt["coordinates"]) + 1)
    high_confidence = wt["plddt_array"] >= 85.0
    for start, end in config["exclude_high_confidence"]:
        high_confidence &= ~((positions >= start) & (positions <= end))
    domain_masks = {
        name: high_confidence & (positions >= start) & ((positions <= end) if end else True)
        for name, (start, end) in config["domains"].items()
    }
    protected_mask = high_confidence & np.asarray([position in protected_positions for position in positions])
    essential_mask = np.asarray([position in essential_positions for position in positions])

    structure_dir = out / "af2_structures"
    structure_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    for name, result in results.items():
        structure = gemmi.read_structure(str(result["pdb_path"]))
        cif_path = structure_dir / f"{name}_af2.cif"
        structure.make_mmcif_document().write_file(str(cif_path))
        if name == "WT":
            continue
        domain_rmsds = {
            domain: kabsch_rmsd(wt["coordinates"], result["coordinates"], mask)
            for domain, mask in domain_masks.items()
        }
        global_rmsd = kabsch_rmsd(wt["coordinates"], result["coordinates"], high_confidence)
        protected_rmsd = kabsch_rmsd(wt["coordinates"], result["coordinates"], protected_mask)
        essential_rmsd = kabsch_rmsd(wt["coordinates"], result["coordinates"], essential_mask)
        esm_path = out / "structures" / f"{name}_esmfold2fast.cif"
        esm_coordinates, esm_plddt = ca_coordinates(esm_path)
        if len(esm_coordinates) != len(result["coordinates"]):
            raise ValueError(f"{name}: AF2 and ESM structure lengths differ")
        cross_mask = high_confidence & (esm_plddt >= 85.0)
        af2_vs_esm_rmsd = kabsch_rmsd(result["coordinates"], esm_coordinates, cross_mask)
        max_domain_rmsd = max(domain_rmsds.values())
        gate_checks = {
            "plddt": result["mean_plddt"] >= wt["mean_plddt"] - config["plddt_tolerance"],
            "ptm": result["ptm"] >= wt["ptm"] - config["ptm_tolerance"],
            "max_domain_rmsd": max_domain_rmsd <= config["max_domain_rmsd"],
            "essential_rmsd": essential_rmsd <= config["essential_rmsd"],
        }
        if config["gate_protected_rmsd"]:
            gate_checks["protected_rmsd"] = protected_rmsd <= config["protected_rmsd"]
        rows.append({
            "design": name,
            "af2_plddt": result["mean_plddt"],
            "af2_plddt_delta_vs_parent": result["mean_plddt"] - wt["mean_plddt"],
            "af2_ptm": result["ptm"],
            "af2_ptm_delta_vs_parent": result["ptm"] - wt["ptm"],
            "af2_global_ca_rmsd_vs_parent_angstrom": global_rmsd,
            "af2_protected_ca_rmsd_vs_parent_angstrom": protected_rmsd,
            "af2_essential_ca_rmsd_vs_parent_angstrom": essential_rmsd,
            "af2_max_domain_ca_rmsd_vs_parent_angstrom": max_domain_rmsd,
            "af2_vs_esmfold2fast_ca_rmsd_angstrom": af2_vs_esm_rmsd,
            **{f"af2_{domain}_ca_rmsd_vs_parent_angstrom": value for domain, value in domain_rmsds.items()},
            "af2_gate_pass": all(gate_checks.values()),
            "af2_failed_gates": ";".join(key for key, value in gate_checks.items() if not value),
            "af2_structure_key": f"af2_structures/{name}_af2.cif",
        })

    rows.sort(key=lambda row: int(row["design"].rsplit("_", 1)[1]))
    by_design = {row["design"]: row for row in rows}
    payload_candidates = payload["candidates"]
    if set(by_design) != {row["design"] for row in payload_candidates}:
        raise ValueError("AF2 outputs and payload candidate names differ")
    for candidate in payload_candidates:
        row = by_design[candidate["design"]]
        candidate["measurements"]["af2_plddt"] = metric(
            row["af2_plddt"], wt["mean_plddt"], "0-1", "higher"
        )
        candidate["measurements"]["af2_ptm"] = metric(
            row["af2_ptm"], wt["ptm"], "0-1", "higher"
        )
        candidate["measurements"]["af2_protected_ca_rmsd"] = metric(
            row["af2_protected_ca_rmsd_vs_parent_angstrom"], 0.0, "angstrom", "lower"
        )
        candidate["measurements"]["af2_essential_ca_rmsd"] = metric(
            row["af2_essential_ca_rmsd_vs_parent_angstrom"], 0.0, "angstrom", "lower"
        )
        candidate["measurements"]["af2_max_domain_ca_rmsd"] = metric(
            row["af2_max_domain_ca_rmsd_vs_parent_angstrom"], 0.0, "angstrom", "lower"
        )
        candidate["af2_validation"] = {
            key: value for key, value in row.items()
            if key not in {"design", "af2_structure_key"}
        }
        candidate["af2_structure_key"] = row["af2_structure_key"]

    payload["schema_version"] = "1.1.0"
    payload["methods"]["af2"] = (
        "local ColabFold 1.6.2 AlphaFold2-pTM model 1; existing target MSA; "
        "3 recycles, 1 ensemble, seed 0, no templates, no relaxation"
    )
    payload["provenance"]["analysis_commit"] = args.analysis_commit
    payload["provenance"]["analysis_commit_url"] = f"{config['repository']}/commit/{args.analysis_commit}"
    payload["parent"]["metrics"]["af2_plddt"] = wt["mean_plddt"]
    payload["parent"]["metrics"]["af2_ptm"] = wt["ptm"]
    payload["parent"]["af2_structure_key"] = "af2_structures/WT_af2.cif"
    payload["summary"]["af2_pass_candidates"] = sum(row["af2_gate_pass"] for row in rows)
    payload["summary"]["af2_fail_candidates"] = sum(not row["af2_gate_pass"] for row in rows)
    payload["summary"]["af2_failed_designs"] = [row["design"] for row in rows if not row["af2_gate_pass"]]
    protected_gate_text = (
        f", {config['protected_label']} RMSD <= {config['protected_rmsd']:.2f} Å"
        if config["gate_protected_rmsd"] else ""
    )
    payload["summary"]["af2_validation_definition"] = (
        f"candidate versus AF2 parent: pLDDT >= parent-{config['plddt_tolerance']:.3f}, "
        f"pTM >= parent-{config['ptm_tolerance']:.3f}, max-domain RMSD <= {config['max_domain_rmsd']:.2f} Å, "
        f"essential-site RMSD <= {config['essential_rmsd']:.2f} Å{protected_gate_text}"
    )

    merged_path = out / f"{args.project}_candidate_measurements_with_af2.json"
    csv_path = out / f"{args.project}_af2_validation.csv"
    summary_path = out / f"{args.project}_af2_validation_summary.json"
    merged_path.write_text(json.dumps(payload, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "project": args.project,
        "method": payload["methods"]["af2"],
        "execution": "local Docker containers on eight aws0 H100 GPUs",
        "image": "ghcr.io/sokrypton/colabfold:1.6.2-cuda12",
        "parent": {"af2_plddt": wt["mean_plddt"], "af2_ptm": wt["ptm"]},
        "candidates": len(rows),
        "gate_pass": sum(row["af2_gate_pass"] for row in rows),
        "gate_fail": sum(not row["af2_gate_pass"] for row in rows),
        "failed_designs": [row["design"] for row in rows if not row["af2_gate_pass"]],
        "ranges": {
            key: [min(float(row[key]) for row in rows), max(float(row[key]) for row in rows)]
            for key in [
                "af2_plddt", "af2_plddt_delta_vs_parent", "af2_ptm",
                "af2_ptm_delta_vs_parent", "af2_global_ca_rmsd_vs_parent_angstrom",
                "af2_protected_ca_rmsd_vs_parent_angstrom",
                "af2_essential_ca_rmsd_vs_parent_angstrom",
                "af2_max_domain_ca_rmsd_vs_parent_angstrom",
                "af2_vs_esmfold2fast_ca_rmsd_angstrom",
            ]
        },
        "outputs": {
            "merged_payload": str(merged_path),
            "csv": str(csv_path),
            "structures": str(structure_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
