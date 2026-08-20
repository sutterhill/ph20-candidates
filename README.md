# PH20 exact ten-mutation redesign campaign

This repository contains the reproducible analysis and final 48-design release
for the soluble human PH20/SPAM1 construct supplied by the user. The construct
is an initiator methionine followed by canonical human PH20 residues 36–483.

The campaign uses local Ridgey v2 600M checkpoints, ESMFold2-Fast structures,
an MMseqs family alignment, and an L2-regularized low-rank Potts model. Catalytic,
substrate-binding, disulfide, N-glycosylation-sequon, terminus, and structure-based
active-pocket neighborhoods are excluded from mutation.

## Protected catalytic mapping

Construct numbering includes the added initiator methionine. Human full-length
PH20 positions therefore map to construct positions by subtracting 34.

- Asp146 → Asp112
- Glu148 → Glu114
- Arg211 → Arg177
- Tyr264 → Tyr230
- Asp284 → Asp250
- Arg287 → Arg253

The design pipeline additionally protects every cysteine and every residue in an
N-X-S/T sequon.
