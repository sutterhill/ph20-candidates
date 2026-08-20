#!/usr/bin/env python3
"""Define conservative no-mutation positions for the soluble PH20 construct."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np


ROOT = Path("/home/ubuntu/codex_ph20_20260820")
STRUCTURE = ROOT / "work/esmfold2fast_parent/cif/PH20_soluble_36_483_construct.cif"
OUTPUT = ROOT / "work/ph20_protected_positions.json"

# Construct numbering: initiator Met followed by human PH20 residues 36–483.
# D112/E114/Y230 form the conserved catalytic machinery. R177/D250/R253 were
# also experimentally essential in PH20 mutagenesis and are treated as active-site
# residues rather than merely conserved positions.
ESSENTIAL = {
    112: "catalytic Asp146 (full-length numbering)",
    114: "catalytic Glu148 proton donor",
    177: "substrate-binding Arg211",
    230: "catalytic Tyr264",
    250: "activity-essential Asp284",
    253: "substrate-binding Arg287",
}


def read_sequence() -> str:
    return "".join(
        line.strip()
        for line in (ROOT / "ph20.fasta").read_text().splitlines()
        if not line.startswith(">")
    )


def main() -> None:
    sequence = read_sequence()
    structure = gemmi.read_structure(str(STRUCTURE))
    chain = structure[0][0]
    residues = []
    observed_pieces: list[str] = []
    for residue in chain:
        amino_acid = gemmi.find_tabulated_residue(residue.name).one_letter_code
        if amino_acid in "ACDEFGHIKLMNPQRSTVWY":
            residues.append(residue)
            observed_pieces.append(amino_acid)
    observed = "".join(observed_pieces)
    if observed != sequence:
        raise ValueError("PH20 structure sequence does not match the supplied construct")

    reasons: dict[int, set[str]] = defaultdict(set)
    for position, description in ESSENTIAL.items():
        reasons[position].add(description)

    # Protect the complete three-residue N-X-S/T sequon because this is a secreted,
    # disulfide-rich glycoprotein and glycan occupancy can affect expression/stability.
    sequons: list[dict] = []
    for index in range(len(sequence) - 2):
        triplet = sequence[index : index + 3]
        if triplet[0] == "N" and triplet[1] != "P" and triplet[2] in "ST":
            positions = [index + 1, index + 2, index + 3]
            sequons.append({"motif": triplet, "positions_1_indexed": positions})
            for position in positions:
                reasons[position].add(f"N-glycosylation sequon {triplet} at {index + 1}")

    # Keep every cysteine and a two-residue local neighborhood unchanged to avoid
    # perturbing extracellular disulfide geometry.
    cysteines = [index + 1 for index, amino_acid in enumerate(sequence) if amino_acid == "C"]
    for cysteine in cysteines:
        for position in range(max(1, cysteine - 2), min(len(sequence), cysteine + 2) + 1):
            reasons[position].add(f"within two residues of Cys{cysteine}")

    # Protect any residue with any atom within 10 Å of an essential-site atom.
    residue_coordinates = [
        np.asarray([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in residue], dtype=float)
        for residue in residues
    ]
    essential_coordinates = np.concatenate(
        [residue_coordinates[position - 1] for position in ESSENTIAL], axis=0
    )
    pocket_positions: set[int] = set()
    for position, coordinates in enumerate(residue_coordinates, 1):
        minimum_distance = float(
            np.linalg.norm(
                coordinates[:, None, :] - essential_coordinates[None, :, :], axis=2
            ).min()
        )
        if minimum_distance <= 10.0:
            pocket_positions.add(position)
            reasons[position].add("within 10 Å of essential catalytic/substrate-binding atoms")

    # Preserve construct termini, which can be sensitive to trimming and secretion.
    for position in list(range(1, 9)) + list(range(len(sequence) - 7, len(sequence) + 1)):
        reasons[position].add("construct terminus")

    output = {
        "construct": "initiator Met + human PH20 residues 36-483",
        "length": len(sequence),
        "essential_positions_1_indexed": ESSENTIAL,
        "cysteine_positions_1_indexed": cysteines,
        "n_glycosylation_sequons": sequons,
        "active_pocket_positions_1_indexed": sorted(pocket_positions),
        "protected_positions_1_indexed": sorted(reasons),
        "reasons_by_position": {
            str(position): sorted(values) for position, values in sorted(reasons.items())
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "length": len(sequence),
        "essential": len(ESSENTIAL),
        "cysteines": len(cysteines),
        "sequons": len(sequons),
        "active_pocket": len(pocket_positions),
        "protected_total": len(reasons),
        "mutable_remaining": len(sequence) - len(reasons),
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
