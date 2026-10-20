#!/usr/bin/env python3

import re
import subprocess
import sys
from pathlib import Path

"""
Filters false positives from all 52 HashFrag jobs using hashFrag-lightning filter_candidates_module.
filter_candidates_module threshold set to 62 (see get_threshold.py).

Also checks for duplicate pairs, if they exist. Combines all pairs in a single table for subsequent filtering.
"""

# ---- Paths and configuration ----
WORK_DIR = Path(
    "/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work"
)
INPUT_DIR = WORK_DIR / "blast_results"
OUTPUT_DIR = WORK_DIR / "blast_filtered_individual"

INPUT_PATTERN = "test_hg38_sequences.part_*.blastn.processed.tsv"
THRESHOLD = "62"

COMBINED_OUTPUT = WORK_DIR / "combined.similar_pairs.tsv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_job_number(input_file: Path) -> str:
    """Extract the eight-digit job number from the input filename."""
    match = re.search(r"\.part_(\d{8})\.blastn\.processed\.tsv$", input_file.name)
    if not match:
        raise ValueError(f"Unexpected input filename: {input_file.name}")
    return match.group(1)


def run_filter(input_file: Path, job_num: str) -> Path:
    """Run HashFrag and rename its output to preserve the job number."""
    # Use a separate output directory for each job to prevent overwrites.
    job_dir = OUTPUT_DIR / f"job_{job_num}"
    job_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "hashFrag",
        "filter_candidates_module",
        "-i", str(input_file),
        "-t", THRESHOLD,
        "-o", str(job_dir),
    ]

    print(f"Running job {job_num}: {input_file.name}", flush=True)
    subprocess.run(cmd, check=True)

    original = job_dir / "hashFrag.similar_pairs.tsv"
    renamed = OUTPUT_DIR / f"hashFrag.similar_pairs_{job_num}.tsv"

    if not original.exists():
        raise FileNotFoundError(
            f"Expected HashFrag output not found: {original}"
        )

    if renamed.exists():
        raise FileExistsError(
            f"Renamed output already exists: {renamed}. "
            "Inspect it before rerunning to avoid overwriting results."
        )

    original.rename(renamed)
    return renamed


def pair_rows(path: Path):
    """Yield each query-matched pair from a TSV, one row at a time."""
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue

            query_id, matched_id, score = line.rstrip("\n").split("\t")
            yield (query_id, matched_id)


def main():
    inputs = sorted(INPUT_DIR.glob(INPUT_PATTERN))

    if not inputs:
        sys.exit(f"No inputs matched: {INPUT_DIR / INPUT_PATTERN}")

    if COMBINED_OUTPUT.exists():
        sys.exit(
            f"{COMBINED_OUTPUT} already exists. Rename or remove it "
            "after checking the previous run."
        )

    # Keep only pair identifiers in memory, not the full result tables.
    seen_pairs = set()
    accepted_count = 0

    with COMBINED_OUTPUT.open("w") as combined:
        for i, input_file in enumerate(inputs, start=1):
            job_num = get_job_number(input_file)
            output_file = run_filter(input_file, job_num)

            # Check this file for pairs already seen in earlier outputs.
            current_pairs = set(pair_rows(output_file))
            overlap = seen_pairs.intersection(current_pairs)

            if overlap:
                print(
                    f"OVERLAP in job {job_num}: "
                    f"{len(overlap)} duplicate query-subject pairs.",
                    flush=True,
                )
                print("Examples:", list(overlap)[:10])
                sys.exit(
                    "Stopped before appending this job. "
                    "Inspect the overlapping pairs before continuing."
                )

            seen_pairs.update(current_pairs)

            with output_file.open("r") as result:
                for line in result:
                    combined.write(line)

            accepted_count += len(current_pairs)
            print(
                f"[{i}/{len(inputs)}] Accepted job {job_num}: "
                f"{len(current_pairs)} pairs; "
                f"{accepted_count} total pairs",
                flush=True,
            )
            print("\n")

    print(f"Combined table: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
