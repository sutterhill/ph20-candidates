#!/usr/bin/env python3
"""Evolutionary-history tools.

Two tools share this file:

1. MSA — a hosted MMseqs endpoint. Needs nothing but network access (no GPU,
   no Modal login). Runs as a plain Python CLI:

    python3 skills/evolutionary-history-tools/scripts/ev_history.py msa \
        --sequence MKTAYIAKQRQISFVKSHFSRQDILD \
        --out msa.a3m

2. EV couplings — our own low-rank Potts fit on the MSA. This actually runs a
   model, so choose local CUDA/CPU or Modal, then invoke the matching command.
   There is no --backend flag; the command you run is the choice:

    # local (auto-detects CUDA, falls back to CPU)
    python3 skills/evolutionary-history-tools/scripts/ev_history.py couplings \
        --sequence MKTAYIAKQRQISFVKSHFSRQDILD \
        --out-prefix ev

    # Modal A100
    modal run skills/evolutionary-history-tools/scripts/ev_history.py::couplings_modal \
        --sequence MKTAYIAKQRQISFVKSHFSRQDILD \
        --out-prefix ev

Outputs from couplings:
  - {out_prefix}.npz:
      frequencies    [L, 20], AA order ACDEFGHIKLMNPQRSTVWY
      couplings      [L, L], APC-corrected coupling scores (contact-map style)
      V              [L, rank, 21], low-rank Potts coupling vectors incl. gap
      h              [L, 21], Potts fields incl. gap
      entropy_bits   [L], per-position Shannon conservation
      gap_fraction   [L], per-position weighted gap frequency
      dE_independent [L, 20], single-mutant log-odds vs. query (site model)
      dE_potts       [L, 20], single-mutant statistical-energy delta vs. query
                     (epistatic Potts model)
      query          the wildtype/query sequence the dE maps are relative to
  - {out_prefix}.summary.json: n_msa, length, n_eff, rank, selected lambda,
      mean_entropy_bits, top_contacts, etc.
  - {out_prefix}.a3m: MSA used for the fit

The entropy/gap/dE/top_contacts fields are example analyses showing common
things you do with an MSA + Potts model; they are not an exhaustive toolkit.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import time
from pathlib import Path

import requests

try:
    import modal
except Exception:  # noqa: BLE001
    modal = None


SUBMIT_URL = "https://shv-internal--mmseqs-msa.modal.run"
RESULT_URL = "https://shv-internal--mmseqs-msa-result.modal.run"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate("-" + AMINO_ACIDS)}
NUM_STATES = 21


# --- sequence + MSA helpers (pure python, no modal/torch) -------------------


def clean_sequence(sequence: str) -> str:
    """Normalize CLI/FASTA text into one canonical amino-acid sequence."""
    if ">" in sequence:
        lines = [ln.strip() for ln in sequence.splitlines() if not ln.startswith(">")]
        sequence = "".join(lines)
    sequence = re.sub(r"\s+", "", sequence).upper()
    if not sequence:
        raise ValueError("empty sequence")
    bad = sorted(set(sequence) - set(AMINO_ACIDS))
    if bad:
        raise ValueError(f"sequence contains non-canonical residues: {''.join(bad)}")
    return sequence


def read_sequence(sequence: str, fasta: str) -> str:
    """Read exactly one sequence from --sequence or --fasta."""
    if bool(sequence) == bool(fasta):
        raise ValueError("pass exactly one of --sequence or --fasta")
    if fasta:
        return clean_sequence(Path(fasta).read_text())
    return clean_sequence(sequence)


def parse_msa_tarball(tarball_bytes: bytes, max_sequences: int = 0) -> list[str]:
    """Extract aligned sequences from the MMseqs result tarball.

    MMseqs/ColabFold A3M uses lowercase letters for insertions relative to the
    query. EV-coupling code needs a rectangular alignment, so lowercase
    insertions are removed. Duplicate rows and rows with unexpected length are
    dropped.
    """
    sequences: list[str] = []
    seen: set[str] = set()

    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".a3m") or member.name.endswith(".dbtype"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            text = fh.read().decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith(">") or line.startswith("#"):
                    continue
                aligned = re.sub(r"[a-z]", "", line).upper()
                aligned = aligned.replace(".", "-")
                if aligned not in seen:
                    seen.add(aligned)
                    sequences.append(aligned)
                    if max_sequences and len(sequences) >= max_sequences:
                        break
            if max_sequences and len(sequences) >= max_sequences:
                break

    if not sequences:
        raise RuntimeError("MSA endpoint returned no aligned sequences")

    target_len = len(sequences[0])
    sequences = [s for s in sequences if len(s) == target_len]
    if not sequences:
        raise RuntimeError("all MSA rows were filtered by length")
    return sequences


def msa_to_a3m(sequences: list[str], name: str = "query") -> str:
    """Write aligned sequences as simple A3M/FASTA text."""
    lines = []
    for i, seq in enumerate(sequences):
        header = name if i == 0 else f"seq_{i}"
        lines.append(f">{header}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


def fetch_msa(sequence: str, poll_seconds: int = 10, max_sequences: int = 0) -> tuple[list[str], str, dict]:
    """Call the shared MMseqs endpoint and return parsed MSA rows."""
    sequence = clean_sequence(sequence)
    resp = requests.post(SUBMIT_URL, json={"input_seqs": [sequence]}, timeout=120)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"[msa] job_id={job_id}", flush=True)

    while True:
        status_resp = requests.get(RESULT_URL, params={"job_id": job_id}, timeout=120)
        status_resp.raise_for_status()
        data = status_resp.json()
        status = data.get("status")
        print(f"[msa] status={status}", flush=True)
        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError(f"MSA job {job_id} failed: {data}")
        time.sleep(poll_seconds)

    result_resp = requests.get(data["presigned_url"], timeout=600)
    result_resp.raise_for_status()

    sequences = parse_msa_tarball(result_resp.content, max_sequences=max_sequences)
    a3m = msa_to_a3m(sequences)
    meta = {
        "job_id": job_id,
        "n_msa": len(sequences),
        "length": len(sequences[0]),
        "s3_path": data.get("s3_path"),
    }
    return sequences, a3m, meta


# --- EV-coupling / low-rank Potts fit (needs numpy + torch) -----------------


def _encode_msa(sequences: list[str], device):
    """Encode MSA strings to integer tensor [N, L]."""
    import numpy as np
    import torch

    lookup = np.zeros(128, dtype=np.int64)
    for char, idx in AA_TO_IDX.items():
        lookup[ord(char)] = idx
    raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    length = len(sequences[0])
    encoded = lookup[raw].reshape(len(sequences), length)
    return torch.from_numpy(encoded).to(device)


def _sequence_weights(x, theta: float = 0.8):
    """Compute sequence reweighting: w_i = 1 / count(identity(i,j) >= theta)."""
    import torch
    import torch.nn.functional as F

    n_seq, length = x.shape
    x_oh = F.one_hot(x, NUM_STATES).float().reshape(n_seq, -1)
    counts = torch.zeros(n_seq, device=x.device)
    chunk = 2048
    for start in range(0, n_seq, chunk):
        sim = (x_oh[start : start + chunk] @ x_oh.T) / length
        counts[start : start + chunk] = (sim >= theta).sum(dim=1).float()
    return 1.0 / counts


def _low_rank_potts_class():
    """Define the model class inside the process after torch import."""
    import torch
    import torch.nn.functional as F

    class LowRankPotts(torch.nn.Module):
        def __init__(self, length: int, q: int = NUM_STATES, rank: int = 32):
            super().__init__()
            self.length = length
            self.q = q
            self.rank = rank
            self.h = torch.nn.Parameter(torch.zeros(length, q))
            self.V = torch.nn.Parameter(torch.randn(length, rank, q) * 0.01)

        def pseudolikelihood(self, x, weights, mask=None):
            n_seq, length = x.shape
            v_perm = self.V.permute(0, 2, 1)
            idx = x.T.unsqueeze(-1).expand(length, n_seq, self.rank)
            v_x = torch.gather(
                v_perm.unsqueeze(1).expand(length, n_seq, self.q, self.rank),
                2,
                idx.unsqueeze(2),
            ).squeeze(2)
            v_x = v_x.permute(1, 0, 2)
            context = v_x.sum(dim=1)
            context_minus = context.unsqueeze(1) - v_x
            logits = torch.einsum("nlr,lrq->nlq", context_minus, self.V) + self.h.unsqueeze(0)

            nll = F.cross_entropy(
                logits.reshape(n_seq * length, self.q),
                x.reshape(n_seq * length),
                reduction="none",
            ).reshape(n_seq, length)

            if mask is not None:
                w = weights.unsqueeze(1) * mask.float()
            else:
                w = weights.unsqueeze(1).expand(n_seq, length)
            return (nll * w).sum() / w.sum()

    return LowRankPotts


def evaluate_msa(
    sequences: list[str],
    rank: int = 32,
    holdout_fraction: float = 0.1,
    lambda_min: float = 1e-4,
    lambda_max: float = 1.0,
    n_lambda: int = 8,
    lr: float = 0.1,
    n_steps: int = 500,
    theta: float = 0.8,
    device_name: str | None = None,
) -> dict:
    """Fit low-rank Potts model and return frequencies/couplings arrays.

    Auto-detects CUDA and falls back to CPU, so this runs locally or on Modal
    with the same code.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    device_name = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[ev] encoding MSA on {device}", flush=True)
    x = _encode_msa(sequences, device)
    n_seq, length = x.shape

    print("[ev] computing sequence weights", flush=True)
    weights = _sequence_weights(x, theta=theta)
    n_eff = float(weights.sum())

    x_oh = F.one_hot(x, NUM_STATES).float()
    freqs = (x_oh * weights[:, None, None]).sum(dim=0)
    freqs = freqs / freqs.sum(dim=1, keepdim=True)

    print(f"[ev] sweeping {n_lambda} lambdas", flush=True)
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    train_mask = torch.rand(n_seq, length, device=device, generator=rng) > holdout_fraction
    val_mask = ~train_mask

    lambdas = torch.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambda).tolist()
    LowRankPotts = _low_rank_potts_class()
    models = [LowRankPotts(length, NUM_STATES, rank).to(device) for _ in lambdas]
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in models]
    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
        for optimizer in optimizers
    ]

    for step in range(n_steps):
        for model, optimizer, scheduler, lam in zip(models, optimizers, schedulers, lambdas):
            optimizer.zero_grad()
            nll = model.pseudolikelihood(x, weights, mask=train_mask)
            reg = lam * (model.V**2).sum() / (length * rank)
            (nll + reg).backward()
            optimizer.step()
            scheduler.step()
        if step % 100 == 0 or step == n_steps - 1:
            print(f"[ev] sweep step {step + 1}/{n_steps}", flush=True)

    best_lambda = lambdas[0]
    best_val_loss = float("inf")
    with torch.no_grad():
        for model, lam in zip(models, lambdas):
            val_loss = model.pseudolikelihood(x, weights, mask=val_mask).item()
            print(f"[ev] lambda={lam:.2e} val_nll={val_loss:.4f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lambda = lam

    print(f"[ev] best lambda={best_lambda:.2e}; refitting full MSA", flush=True)
    refit_steps = n_steps * 2
    model = LowRankPotts(length, NUM_STATES, rank).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=refit_steps)

    for step in range(refit_steps):
        optimizer.zero_grad()
        nll = model.pseudolikelihood(x, weights)
        reg = best_lambda * (model.V**2).sum() / (length * rank)
        loss = nll + reg
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == refit_steps - 1:
            print(f"[ev] refit step {step + 1}/{refit_steps}: nll={nll.item():.4f}", flush=True)

    print("[ev] computing APC-corrected coupling scores", flush=True)
    with torch.no_grad():
        v = model.V.detach()
        g = torch.einsum("lra,lsa->lrs", v, v)
        g_flat = g.reshape(length, -1)
        gram = g_flat @ g_flat.T
        gram.clamp_(min=0)
        fn_scores = gram.sqrt()
        fn_scores.fill_diagonal_(0)

        col_mean = fn_scores.sum(dim=1) / (length - 1)
        total_mean = col_mean.sum() / (length - 1)
        apc = torch.outer(col_mean, col_mean) / (total_mean + 1e-10)
        cn_scores = fn_scores - apc
        cn_scores.fill_diagonal_(0)

    freqs_aa = freqs[:, 1:].cpu().numpy()
    row_sums = freqs_aa.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    freqs_aa = freqs_aa / row_sums

    # --- example analyses derived from the same fit (cheap extras) ----------
    # These show a few common things you do with an MSA + Potts model. They are
    # examples, not an exhaustive toolkit.
    h = model.h.detach()  # [L, 21] Potts fields; needed for variant scoring
    x0 = x[0]  # query (wildtype) row, integer states including gap

    # 1. Per-position conservation: Shannon entropy (bits) over the 20 AAs.
    faa = freqs[:, 1:]
    faa = faa / faa.sum(dim=1, keepdim=True).clamp(min=1e-9)
    entropy_bits = -(faa * torch.log2(faa.clamp(min=1e-9))).sum(dim=1)  # [L]
    gap_fraction = freqs[:, 0]  # [L], weighted gap frequency per position

    # 2. Single-mutant effect maps relative to the query, over 20 AAs ([L, 20]).
    #    Independent-site log-odds: log f(mut) - log f(wt).
    logf = torch.log(freqs.clamp(min=1e-9))  # [L, 21]
    wt_logf = logf.gather(1, x0[:, None]).squeeze(1)  # [L]
    de_independent = (logf - wt_logf[:, None])[:, 1:]  # [L, 20]

    #    Epistatic Potts statistical-energy delta dE(pos, wt -> state):
    #    dE = (h[s]-h[wt]) + (<V[s], ctx> - <V[wt], ctx>), ctx excludes self.
    rank_dim = v.shape[1]
    v_wt = v.gather(2, x0[:, None, None].expand(length, rank_dim, 1)).squeeze(2)  # [L, rank]
    context = v_wt.sum(dim=0)[None, :] - v_wt  # [L, rank], leave-one-out context
    pe = torch.einsum("lrs,lr->ls", v, context)  # [L, 21]
    pe_wt = pe.gather(1, x0[:, None]).squeeze(1)  # [L]
    h_wt = h.gather(1, x0[:, None]).squeeze(1)  # [L]
    de_potts = ((h - h_wt[:, None]) + (pe - pe_wt[:, None]))[:, 1:]  # [L, 20]

    return {
        "frequencies": freqs_aa.astype("float32"),
        "couplings": cn_scores.cpu().numpy().astype("float32"),
        "V": v.cpu().numpy().astype("float32"),
        "h": h.cpu().numpy().astype("float32"),
        "entropy_bits": entropy_bits.cpu().numpy().astype("float32"),
        "gap_fraction": gap_fraction.cpu().numpy().astype("float32"),
        "dE_potts": de_potts.cpu().numpy().astype("float32"),
        "dE_independent": de_independent.cpu().numpy().astype("float32"),
        "query": sequences[0],
        "N_eff": n_eff,
        "best_lambda": float(best_lambda),
        "best_val_loss": float(best_val_loss),
        "n_msa": int(n_seq),
        "length": int(length),
        "rank": int(rank),
        "theta": float(theta),
    }


def compute_couplings(
    sequence: str,
    *,
    poll_seconds: int = 10,
    max_sequences: int = 0,
    rank: int = 32,
    holdout_fraction: float = 0.1,
    lambda_min: float = 1e-4,
    lambda_max: float = 1.0,
    n_lambda: int = 8,
    lr: float = 0.1,
    n_steps: int = 500,
    theta: float = 0.8,
) -> dict:
    """Fetch the MSA, fit EV couplings, and pack outputs. Used by both the local
    CLI and the Modal worker, so the fit is identical either place."""
    import numpy as np

    msa_rows, a3m, msa_meta = fetch_msa(
        sequence, poll_seconds=poll_seconds, max_sequences=max_sequences
    )
    ev = evaluate_msa(
        msa_rows,
        rank=rank,
        holdout_fraction=holdout_fraction,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        n_lambda=n_lambda,
        lr=lr,
        n_steps=n_steps,
        theta=theta,
    )

    # Example: top predicted contacts = largest APC-corrected couplings with a
    # sequence separation of >= 5, which is how coupling scores are read as a
    # contact map.
    couplings = ev["couplings"]
    length = ev["length"]
    top_contacts = []
    for i in range(length):
        for j in range(i + 5, length):
            top_contacts.append((float(couplings[i, j]), i, j))
    top_contacts.sort(reverse=True)
    top_contacts = [
        {"i": i, "j": j, "score": round(score, 4)}
        for score, i, j in top_contacts[:length]
    ]

    summary = {
        **msa_meta,
        "n_eff": ev["N_eff"],
        "rank": ev["rank"],
        "theta": ev["theta"],
        "best_lambda": ev["best_lambda"],
        "best_val_loss": ev["best_val_loss"],
        "aa_order": AMINO_ACIDS,
        "mean_entropy_bits": float(np.mean(ev["entropy_bits"])),
        "top_contacts": top_contacts,
    }

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        frequencies=ev["frequencies"],
        couplings=ev["couplings"],
        V=ev["V"],
        h=ev["h"],
        entropy_bits=ev["entropy_bits"],
        gap_fraction=ev["gap_fraction"],
        dE_potts=ev["dE_potts"],
        dE_independent=ev["dE_independent"],
        query=np.asarray(ev["query"]),
        aa_order=np.asarray(AMINO_ACIDS),
        n_eff=np.asarray(ev["N_eff"], dtype=np.float32),
        best_lambda=np.asarray(ev["best_lambda"], dtype=np.float32),
    )
    return {"npz": buf.getvalue(), "a3m": a3m, "summary": summary}


def write_couplings_outputs(result: dict, out_prefix: str) -> None:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".npz").write_bytes(result["npz"])
    prefix.with_suffix(".a3m").write_text(result["a3m"])
    prefix.with_suffix(".summary.json").write_text(json.dumps(result["summary"], indent=2))
    for suffix in (".npz", ".a3m", ".summary.json"):
        print(f"[out] wrote {prefix.with_suffix(suffix).resolve()}", flush=True)


# --- Modal entrypoint (only the couplings fit; MSA needs no Modal) ----------

if modal is not None:
    app = modal.App("evolutionary-history-tools")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("requests", "numpy", "torch")
    )

    @app.function(image=image, gpu="A100", cpu=8, memory=65536, timeout=7200)
    def couplings_remote(params: dict) -> dict:
        return compute_couplings(**params)

    @app.local_entrypoint()
    def couplings_modal(
        sequence: str = "",
        fasta: str = "",
        out_prefix: str = "ev",
        poll_seconds: int = 10,
        max_sequences: int = 0,
        rank: int = 32,
        holdout_fraction: float = 0.1,
        lambda_min: float = 1e-4,
        lambda_max: float = 1.0,
        n_lambda: int = 8,
        lr: float = 0.1,
        n_steps: int = 500,
        theta: float = 0.8,
    ):
        """Run the EV-coupling fit on a Modal A100."""
        params = {
            "sequence": read_sequence(sequence, fasta),
            "poll_seconds": poll_seconds,
            "max_sequences": max_sequences,
            "rank": rank,
            "holdout_fraction": holdout_fraction,
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "n_lambda": n_lambda,
            "lr": lr,
            "n_steps": n_steps,
            "theta": theta,
        }
        result = couplings_remote.remote(params)
        write_couplings_outputs(result, out_prefix)


# --- Local CLI --------------------------------------------------------------


def run_msa(args: argparse.Namespace) -> None:
    seq = read_sequence(args.sequence, args.fasta)
    _, a3m, meta = fetch_msa(seq, poll_seconds=args.poll_seconds, max_sequences=args.max_sequences)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(a3m)
    print(f"[out] wrote {out_path.resolve()}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


def run_couplings(args: argparse.Namespace) -> None:
    seq = read_sequence(args.sequence, args.fasta)
    result = compute_couplings(
        seq,
        poll_seconds=args.poll_seconds,
        max_sequences=args.max_sequences,
        rank=args.rank,
        holdout_fraction=args.holdout_fraction,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        n_lambda=args.n_lambda,
        lr=args.lr,
        n_steps=args.n_steps,
        theta=args.theta,
    )
    write_couplings_outputs(result, args.out_prefix)


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("msa", help="Fetch an MSA from the shared endpoint (no GPU/Modal needed).")
    m.add_argument("--sequence", default="")
    m.add_argument("--fasta", default="")
    m.add_argument("--out", default="msa.a3m")
    m.add_argument("--poll-seconds", type=int, default=10)
    m.add_argument("--max-sequences", type=int, default=0, help="0 keeps all parsed rows.")
    m.set_defaults(func=run_msa)

    c = sub.add_parser("couplings", help="Fetch MSA and fit EV couplings locally (auto CUDA/CPU).")
    c.add_argument("--sequence", default="")
    c.add_argument("--fasta", default="")
    c.add_argument("--out-prefix", default="ev")
    c.add_argument("--poll-seconds", type=int, default=10)
    c.add_argument("--max-sequences", type=int, default=0, help="0 keeps all parsed rows.")
    c.add_argument("--rank", type=int, default=32)
    c.add_argument("--holdout-fraction", type=float, default=0.1)
    c.add_argument("--lambda-min", type=float, default=1e-4)
    c.add_argument("--lambda-max", type=float, default=1.0)
    c.add_argument("--n-lambda", type=int, default=8)
    c.add_argument("--lr", type=float, default=0.1)
    c.add_argument("--n-steps", type=int, default=500)
    c.add_argument("--theta", type=float, default=0.8)
    c.set_defaults(func=run_couplings)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    cli_main()
