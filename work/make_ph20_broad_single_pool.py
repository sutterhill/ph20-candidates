#!/usr/bin/env python3
"""Build a conservative, evolution-supported PH20 single-mutant proposal pool."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
MODEL_NAMES = ("base", "ens1", "ens2", "ens3", "ens4")
AA = "ACDEFGHIKLMNPQRSTVWY"
MAX_PER_POSITION = 5
MAX_TOTAL = 800
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}
HYDROPATHY = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
    "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}


def read_a3m(path: Path) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if current:
                rows.append(re.sub("[a-z.]", "", "".join(current)))
                current = []
        else:
            current.append(line.strip())
    if current:
        rows.append(re.sub("[a-z.]", "", "".join(current)))
    return rows


def read_sequence() -> str:
    return "".join(
        line.strip() for line in (ROOT / "ph20.fasta").read_text().splitlines()
        if not line.startswith(">")
    )


def main() -> None:
    sequence = read_sequence()
    parents = [
        json.loads((ROOT / f"work/ridgey_parent_ensemble/{name}.json").read_text())
        for name in MODEL_NAMES
    ]
    if any(parent["sequence"] != sequence for parent in parents):
        raise ValueError("Ridgey parent sequence mismatch")
    predictions = [parent["predictions"] for parent in parents]
    plddt = np.asarray(parents[0]["structure_confidence"]["plddt"], dtype=float)
    msa = [row for row in read_a3m(ROOT / "work/ph20_ev.a3m") if len(row) == len(sequence)]
    protected = set(
        json.loads((ROOT / "work/ph20_protected_positions.json").read_text())[
            "protected_positions_1_indexed"
        ]
    )
    ev = np.load(ROOT / "work/ph20_ev.npz")
    ev_aa = str(ev["aa_order"])

    # Biopython's SASA implementation expects an occupancy column, which the
    # ESMFold2-Fast mmCIF omits. Gemmi safely converts it to PDB first.
    cif_path = ROOT / "work/esmfold2fast_parent/cif/PH20_soluble_36_483_construct.cif"
    pdb_path = ROOT / "work/ph20_parent_esmfold2fast.pdb"
    gemmi.read_structure(str(cif_path)).write_pdb(str(pdb_path))
    structure = PDBParser(QUIET=True).get_structure("ph20", str(pdb_path))
    ShrakeRupley().compute(structure, level="R")
    residues = [residue for residue in next(structure.get_chains()) if residue.id[0] == " "]
    if len(residues) != len(sequence):
        raise ValueError("SASA structure length mismatch")
    blosum = substitution_matrices.load("BLOSUM62")

    candidates: list[dict] = []
    for index, native in enumerate(sequence):
        position = index + 1
        if position in protected or plddt[index] < 85.0 or native in "CPG":
            continue
        counts = Counter(row[index] for row in msa if row[index] in AA)
        non_gap = sum(counts.values())
        if non_gap < 250:
            continue
        native_frequency = counts[native] / non_gap
        relative_sasa = min(2.0, float(residues[index].sasa / MAX_ASA[native]))
        max_ptm = max(float(prediction["ptm"]["any_probability"][index]) for prediction in predictions)
        if max_ptm >= 0.25:
            continue
        for mutant in AA:
            if mutant == native or mutant in "CPG":
                continue
            frequency = counts[mutant] / non_gap
            if frequency < 0.0025:
                continue
            blosum_score = float(blosum[native, mutant])
            if blosum_score < -2 and not (frequency >= 0.08 and frequency >= native_frequency):
                continue
            if relative_sasa > 0.35 and HYDROPATHY[mutant] > HYDROPATHY[native] + 1.0:
                continue
            if relative_sasa < 0.20 and mutant in "DEKR" and frequency < 0.10:
                continue

            stability_deltas: list[float] = []
            lm_deltas: list[float] = []
            for prediction in predictions:
                ddg_order = prediction["ddg_amino_acid_order"]
                lm_order = prediction["lm_amino_acid_order"]
                stability_deltas.append(
                    float(prediction["ddg"][index][ddg_order.index(native)])
                    - float(prediction["ddg"][index][ddg_order.index(mutant)])
                )
                lm_deltas.append(
                    float(prediction["lm"][index][lm_order.index(mutant)])
                    - float(prediction["lm"][index][lm_order.index(native)])
                )
            mean_stability = statistics.mean(stability_deltas)
            mean_lm = statistics.mean(lm_deltas)
            independent = float(ev["dE_independent"][index, ev_aa.index(mutant)])
            potts = float(ev["dE_potts"][index, ev_aa.index(mutant)])
            if min(stability_deltas) < -0.50 or mean_stability < -0.10:
                continue
            if sum(value > 0 for value in lm_deltas) < 2 or mean_lm < -1.0:
                continue
            if independent < -4.0 or potts < -10.0:
                continue
            ranking_score = (
                0.95 * mean_stability
                + 0.18 * mean_lm
                + 0.055 * potts
                + 0.25 * math.log10(1.0 + len(msa) * frequency)
                + 0.05 * blosum_score
                + 0.10 * max(0.0, relative_sasa - 0.35) * max(0.0, HYDROPATHY[native] - HYDROPATHY[mutant])
            )
            candidates.append({
                "mutation": f"{native}{position}{mutant}",
                "position": position,
                "native": native,
                "mutant": mutant,
                "msa_frequency": frequency,
                "native_frequency": native_frequency,
                "blosum62": blosum_score,
                "relative_sasa": relative_sasa,
                "ridgey_ddg_mean": mean_stability,
                "ridgey_ddg_min": min(stability_deltas),
                "ridgey_ddg_positive_models": sum(value > 0 for value in stability_deltas),
                "ridgey_lm_mean": mean_lm,
                "ridgey_lm_min": min(lm_deltas),
                "potts_single_delta": potts,
                "independent_delta": independent,
                "ranking_score": ranking_score,
            })

    by_position: dict[int, list[dict]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda row: row["ranking_score"], reverse=True):
        by_position[candidate["position"]].append(candidate)
    selected = [row for rows in by_position.values() for row in rows[:MAX_PER_POSITION]]
    selected = sorted(selected, key=lambda row: row["ranking_score"], reverse=True)[:MAX_TOTAL]
    if not selected:
        raise RuntimeError("no single-mutant candidates survived")

    csv_path = ROOT / "work/ph20_broad_single_pool.csv"
    fasta_path = ROOT / "work/ph20_broad_single_pool.fasta"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    with fasta_path.open("w") as handle:
        for row in selected:
            position = int(row["position"])
            mutant_sequence = sequence[: position - 1] + row["mutant"] + sequence[position:]
            handle.write(f">{row['mutation']}\n{mutant_sequence}\n")
    print(json.dumps({
        "msa_rows": len(msa),
        "untrimmed_candidates": len(candidates),
        "positions": len(by_position),
        "selected": len(selected),
        "protected_positions": len(protected),
        "csv": str(csv_path),
        "fasta": str(fasta_path),
    }, indent=2))


if __name__ == "__main__":
    main()
