#!/usr/bin/env python3
"""Merge Ridgey/Potts/ESMFold2-Fast results and package 48 PH20 designs."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
FOLD_DIR = ROOT / "work/esmfold2fast_fold120_v2"
OUT_DIR = ROOT / "outputs/ph20_48_10mut_designs"
N_FINAL = 48
AUDIT_EXCLUDED_MUTATIONS = {"Q350R"}


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


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask & np.isfinite(reference).all(axis=1) & np.isfinite(mobile).all(axis=1)
    ref = reference[valid]
    mob = mobile[valid]
    ref0 = ref - ref.mean(axis=0)
    mob0 = mob - mob.mean(axis=0)
    u, _, vt = np.linalg.svd(mob0.T @ ref0)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    distances = np.linalg.norm(mob0 @ rotation - ref0, axis=1)
    return float(np.sqrt(np.mean(distances ** 2))), float(np.mean(distances <= 3.0))


def well_for_rank(rank: int) -> str:
    index = rank - 1
    return f"{chr(ord('A') + index // 8)}{index % 8 + 1}"


def main() -> None:
    sequences = read_fasta(ROOT / "work/ph20_fold128_prefold.fasta")
    prefold = {
        row["name"]: row
        for row in csv.DictReader((ROOT / "work/ph20_fold128_prefold.csv").open())
    }
    folds: dict[str, dict] = {}
    for path in sorted(FOLD_DIR.glob("shard_*.json")):
        for row in json.loads(path.read_text()):
            folds[row["name"]] = row
    if len(folds) != len(sequences):
        raise RuntimeError(f"expected {len(sequences)} fold records, found {len(folds)}")

    wt_fold = folds["WT"]
    wt_coordinates, wt_plddt = ca_coordinates(Path(wt_fold["cif"]))
    if len(wt_coordinates) != 449:
        raise ValueError("unexpected parent structure length")
    positions = np.arange(1, len(wt_coordinates) + 1)
    high_confidence = np.isfinite(wt_plddt) & (wt_plddt >= 85.0)
    protected = json.loads((ROOT / "work/ph20_protected_positions.json").read_text())
    pocket_positions = set(protected["active_pocket_positions_1_indexed"])
    pocket_mask = high_confidence & np.asarray([position in pocket_positions for position in positions])
    domains = {
        "gh56": high_confidence & (positions <= 335),
        "c_terminal": high_confidence & (positions >= 336),
        "active_pocket": pocket_mask,
    }

    evaluated: list[dict] = []
    for source_name, row in prefold.items():
        fold = folds[source_name]
        coordinates, _ = ca_coordinates(Path(fold["cif"]))
        global_rmsd, global_fraction = kabsch_rmsd(wt_coordinates, coordinates, high_confidence)
        domain_rmsds = {
            name: kabsch_rmsd(wt_coordinates, coordinates, mask)[0]
            for name, mask in domains.items()
        }
        plddt_delta = float(fold["plddt"]) - float(wt_fold["plddt"])
        ptm_delta = float(fold["ptm"]) - float(wt_fold["ptm"])
        fold_score = (
            float(row["selection_score"])
            + 12.0 * plddt_delta
            + 8.0 * ptm_delta
            - 0.10 * domain_rmsds["active_pocket"]
            - 0.05 * max(domain_rmsds["gh56"], domain_rmsds["c_terminal"])
        )
        evaluated.append({
            **row,
            "source_name": source_name,
            "sequence": sequences[source_name],
            "esmfold2fast_plddt": float(fold["plddt"]),
            "esmfold2fast_plddt_delta_vs_wt": plddt_delta,
            "esmfold2fast_ptm": float(fold["ptm"]),
            "esmfold2fast_ptm_delta_vs_wt": ptm_delta,
            "structured_ca_rmsd_vs_wt_angstrom": global_rmsd,
            "structured_ca_fraction_within_3A": global_fraction,
            "gh56_ca_rmsd_angstrom": domain_rmsds["gh56"],
            "c_terminal_ca_rmsd_angstrom": domain_rmsds["c_terminal"],
            "active_pocket_ca_rmsd_angstrom": domain_rmsds["active_pocket"],
            "max_domain_ca_rmsd_angstrom": max(domain_rmsds["gh56"], domain_rmsds["c_terminal"]),
            "fold_selection_score": fold_score,
            "source_cif": fold["cif"],
        })

    protected_positions = set(protected["protected_positions_1_indexed"])
    passing = []
    for row in evaluated:
        mutation_positions = [int(mutation[1:-1]) for mutation in row["mutations"].split(";")]
        if any(position in protected_positions for position in mutation_positions):
            raise ValueError(f"{row['source_name']} touches a protected position")
        if (
            row["esmfold2fast_plddt"] >= float(wt_fold["plddt"]) - 0.008
            and row["esmfold2fast_ptm"] >= float(wt_fold["ptm"]) - 0.020
            and row["active_pocket_ca_rmsd_angstrom"] <= 0.75
            and row["max_domain_ca_rmsd_angstrom"] <= 1.50
        ):
            passing.append(row)

    selected: list[dict] = []
    mutation_use: Counter[str] = Counter()
    for row in sorted(passing, key=lambda item: item["fold_selection_score"], reverse=True):
        mutations = set(row["mutations"].split(";"))
        if mutations & AUDIT_EXCLUDED_MUTATIONS:
            continue
        if any(len(mutations & set(other["mutations"].split(";"))) > 8 for other in selected):
            continue
        if any(mutation_use[mutation] >= 45 for mutation in mutations):
            continue
        selected.append(row)
        mutation_use.update(mutations)
        if len(selected) == N_FINAL:
            break
    if len(selected) < N_FINAL:
        raise RuntimeError(f"only {len(selected)} diverse designs selected from {len(passing)} fold-passing")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    structures_dir = OUT_DIR / "structures"
    structures_dir.mkdir(parents=True)
    shutil.copy2(wt_fold["cif"], structures_dir / "WT_esmfold2fast.cif")

    fasta_path = OUT_DIR / "ph20_48_10mut_designs.fasta"
    csv_path = OUT_DIR / "ph20_48_10mut_designs.csv"
    manifest_path = OUT_DIR / "ph20_48_manifest.json"
    final_rows: list[dict] = []
    with fasta_path.open("w") as fasta_handle:
        for rank, row in enumerate(selected, 1):
            final_name = f"PH20_10M_{rank:03d}"
            mutations = row["mutations"].split(";")
            if len(mutations) != 10:
                raise ValueError(f"{row['source_name']} does not have ten mutations")
            if sum(a != b for a, b in zip(sequences["WT"], row["sequence"])) != 10:
                raise ValueError(f"{row['source_name']} sequence is not an exact ten-mutant")
            well = well_for_rank(rank)
            fasta_handle.write(f">{final_name} well={well} mutations={'|'.join(mutations)}\n{row['sequence']}\n")
            destination = structures_dir / f"{final_name}_esmfold2fast.cif"
            shutil.copy2(row["source_cif"], destination)
            final_rows.append({
                "design": final_name,
                "well": well,
                **{key: value for key, value in row.items() if key not in {"sequence", "source_cif", "name"}},
                "structure_file": f"structures/{destination.name}",
            })

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)

    manifest = {
        "target": "soluble human PH20/SPAM1; initiator Met + UniProt P38567 residues 36-483",
        "wild_type_length": len(sequences["WT"]),
        "generated_unique_exact_ten_mutants": sum(1 for _ in (ROOT / "work/ph20_generated_10mut_pool.csv").open()) - 1,
        "five_model_scored": 400,
        "folded": len(evaluated),
        "fold_passing": len(passing),
        "design_count": len(final_rows),
        "mutations_per_design": 10,
        "protected_essential_construct_positions": [112, 114, 177, 230, 250, 253],
        "protected_positions_total": len(protected_positions),
        "ridgey_models": ["600m", "600m_ens1", "600m_ens2", "600m_ens3", "600m_ens4"],
        "potts": {
            "msa_sequences": 3801,
            "n_eff": 1336.0673828125,
            "rank": 32,
            "selected_l2_lambda": 1.0,
        },
        "esmfold2fast_wt": {"plddt": wt_fold["plddt"], "ptm": wt_fold["ptm"]},
        "selection": {
            "stability": "positive delta in all five Ridgey 600M checkpoints",
            "ec": "positive ensemble-mean P(EC 3.2.1.35), non-decreasing in >=4/5 checkpoints, worst delta >= -0.0018",
            "solubility": "positive ensemble-mean delta",
            "sequence_likelihood": "positive ensemble-mean mutation-site masked log-probability delta",
            "fold": "ESMFold2-Fast pLDDT >= WT-0.008, pTM >= WT-0.020, active-pocket RMSD <=0.75 Å, domain RMSD <=1.50 Å",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    report_path = OUT_DIR / "README.md"
    report_path.write_text(
        "# PH20 48 × exact ten-mutation candidates\n\n"
        "These are computationally prioritized candidates, not experimentally validated proteins. "
        "The parent is an initiator methionine followed by human PH20 residues 36–483.\n\n"
        f"The campaign generated {manifest['generated_unique_exact_ten_mutants']:,} unique exact ten-mutants, "
        f"five-model scored {manifest['five_model_scored']}, folded {manifest['folded']}, and retained "
        f"{manifest['fold_passing']} through structural gates before selecting 48 diverse finalists.\n\n"
        "No finalist changes a catalytic/substrate residue, any residue within the 10 Å protected active-pocket shell, "
        "a cysteine/disulfide neighborhood, or an N-glycosylation sequon.\n"
    )

    archive_path = OUT_DIR / "ph20_48_structures.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(structures_dir.glob("*.cif")):
            archive.write(path, arcname=path.name)
    print(json.dumps({
        "folded": len(evaluated),
        "fold_passing": len(passing),
        "selected": len(final_rows),
        "output_dir": str(OUT_DIR),
        "wt_plddt": wt_fold["plddt"],
        "wt_ptm": wt_fold["ptm"],
        "final_stability_delta_mean_range": [
            min(float(row["ridgey_stability_delta_mean"]) for row in final_rows),
            max(float(row["ridgey_stability_delta_mean"]) for row in final_rows),
        ],
        "final_ec_delta_mean_range": [
            min(float(row["ridgey_ec_delta_mean"]) for row in final_rows),
            max(float(row["ridgey_ec_delta_mean"]) for row in final_rows),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
