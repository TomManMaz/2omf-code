#!/usr/bin/env python3
"""Post-processing pipeline for 2-OMF experimental results.

Merges scripts 0–6 from the lab repo into a single reproducible pipeline:
  Step 0: Extract results.tar.gz
  Step 1: Parse stderr logs from algorithm runs
  Step 2: Extract lower bounds from Gurobi logs + static bounds from instances
  Step 3: Merge per-algorithm summaries into a single CSV
  Step 4: Create LaTeX tables (pivoted results, gap tables)
  Step 5: Create plots (violin, scalability, boxplots)
  Step 6: Compute win/loss/draw statistics
  Step 7: Calculate and export gap statistics
  Step 8: Generate CSVs for statistical evaluation (R script)

Usage:
    python 2-postprocessing.py              # run full pipeline
    python 2-postprocessing.py --from-step 4  # skip parsing, start from tables
"""

import argparse
import concurrent.futures
import logging
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_DIR / "results"
INSTANCES_DIR = REPO_DIR / "instances"
OUTPUT_DIR = REPO_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

ALGO_DIR_TO_NAME = {"sa": "simulated_annealing", "hg2": "hg2", "gurobi": "gurobi"}

ALGO_DISPLAY = {"gurobi": "GRB", "simulated_annealing": "SA", "hg2": "HG2"}

ALGORITHM_COLUMNS = ["simulated_annealing_mean", "gurobi_mean", "hg2_mean"]
ALGORITHM_DISPLAY_NAMES_TABLE = ["SA", "Gurobi", "{HG2}"]

GAP_SCALE = 1e3
GAP_COLUMN_ORDER = ["simulated_annealing", "gurobi", "hg2"]
GAP_DISPLAY_NAMES = {"simulated_annealing": "SA", "gurobi": "Gurobi", "hg2": "HG2"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
)
logger = logging.getLogger("postprocessing")

# ---------------------------------------------------------------------------
# Plotting constants (lazy-imported)
# ---------------------------------------------------------------------------

_ALGO_COLORS: dict | None = None


def _get_algo_colors() -> dict:
    global _ALGO_COLORS
    if _ALGO_COLORS is None:
        import seaborn as sns

        _palette = sns.color_palette("colorblind")
        _ALGO_COLORS = {"GRB": _palette[0], "SA": _palette[1], "HG2": _palette[2]}
    return _ALGO_COLORS


def _rename_algorithm(name: str) -> str:
    return ALGO_DISPLAY.get(name, name)


# ===================================================================
# Step 0 — Extract results
# ===================================================================


def extract_results() -> None:
    """Extract results.tar.gz into results/ if not already present."""
    archive = REPO_DIR / "results.tar.gz"
    if RESULTS_DIR.is_dir() and any(RESULTS_DIR.rglob("stderr.log")):
        logger.info("results/ already contains log files — skipping extraction.")
        return
    if not archive.exists():
        logger.error("results.tar.gz not found in %s", REPO_DIR)
        sys.exit(1)
    logger.info("Extracting results.tar.gz …")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(REPO_DIR, filter="data")
    count = sum(1 for _ in RESULTS_DIR.rglob("stderr.log"))
    logger.info("Extracted %d stderr.log files into results/", count)


# ===================================================================
# Step 1 — Parse stderr logs  (from 0-stderr_checker.py)
# ===================================================================


@dataclass
class OutputFormat:
    objective_function_value: Optional[float]
    iterations: Optional[float]
    improvements: Optional[float]
    duration_seconds: Optional[float]
    seed: Optional[int]
    instance: Optional[str] = None


def _parse_output(content: str) -> Optional[OutputFormat]:
    ofv_m = re.search(r"Objective function value:\s*([\-\d.]+)", content)
    it_m = re.search(r"Iterations:\s*([\-\d.]+)", content)
    imp_m = re.search(r"Improvements:\s*([\-\d.]+)", content)
    dur_m = re.search(r"Duration:\s*([\-\d.]+)\s*seconds", content)
    seed_m = re.search(r"seed\s*=\s*(\d+)", content)
    instance_m = re.search(r"([A-Za-z0-9_.-]+_\d[A-Za-z0-9_.-]*\.dat)", content)

    if not (ofv_m or it_m or imp_m or dur_m or seed_m):
        return None

    def to_float(m):
        try:
            return float(m.group(1)) if m else None
        except Exception:
            return None

    def to_int(m):
        try:
            return int(m.group(1)) if m else None
        except Exception:
            return None

    inst = None
    if instance_m:
        name = instance_m.group(1)
        inst = name[:-4] if name.lower().endswith(".dat") else name

    return OutputFormat(
        objective_function_value=to_float(ofv_m),
        iterations=to_float(it_m),
        improvements=to_float(imp_m),
        duration_seconds=to_float(dur_m),
        seed=to_int(seed_m),
        instance=inst,
    )


def _determine_status(content: str) -> str:
    if "Completely finished after" in content:
        return "completed"
    if "KeyboardInterrupt" in content:
        return "interrupted"
    return "failed"


def _process_single_stderr(file_path: Path, base_dir: Path) -> dict:
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        content = ""

    status = _determine_status(content)
    output = _parse_output(content)

    # Derive instance name from path as fallback
    instance_from_path = None
    try:
        rel = file_path.relative_to(base_dir)
        parts = rel.parts
        if parts:
            instance_from_path = parts[0]
    except Exception:
        pass

    instance_value = (output.instance if output and output.instance else instance_from_path)

    return {
        "Instances": instance_value,
        "file_path": str(file_path),
        "status": status,
        "objective_function_value": output.objective_function_value if output else None,
        "iterations": output.iterations if output else None,
        "improvements": output.improvements if output else None,
        "duration_seconds": output.duration_seconds if output else None,
    }


def parse_stderr_logs() -> None:
    """Parse stderr.log files for each algorithm and save per-algorithm CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dir_name, algo_name in ALGO_DIR_TO_NAME.items():
        algo_dir = RESULTS_DIR / dir_name
        if not algo_dir.is_dir():
            logger.warning("Directory %s not found — skipping %s", algo_dir, algo_name)
            continue

        stderr_files = sorted(algo_dir.rglob("stderr.log"))
        total = len(stderr_files)
        logger.info("[%s] Found %d stderr.log files", algo_name, total)
        if total == 0:
            continue

        cpu = os.cpu_count() or 1
        max_workers = min(64, max(4, cpu * 4))
        records: list[dict] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_process_single_stderr, p, algo_dir): p
                for p in stderr_files
            }
            processed = 0
            for fut in concurrent.futures.as_completed(futures):
                processed += 1
                try:
                    record = fut.result()
                except Exception:
                    logger.error("Worker exception", exc_info=True)
                    continue
                record["algorithm"] = algo_name
                records.append(record)
                if processed % 200 == 0 or processed == total:
                    logger.info("[%s] Processed %d/%d", algo_name, processed, total)

        csv_path = OUTPUT_DIR / f"{dir_name}_stderr_summary.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        logger.info("Saved %s (%d records)", csv_path.name, len(records))


# ===================================================================
# Step 2 — Extract lower bounds  (from 1-lower_bound_fetcher.py)
# ===================================================================


def _parse_instance_name_from_log(content: str) -> Optional[str]:
    match = re.search(r"instance_file='[^']*?/([^/']+)\.dat'", content)
    return match.group(1) if match else None


def _parse_lb_stderr(content: str) -> Optional[float]:
    match = re.search(r"Lower bound:\s*([\d.]+)", content)
    return float(match.group(1)) if match else None


def _parse_lb_stdout(stdout_path: Path) -> Optional[float]:
    if not stdout_path.exists():
        return None
    content = stdout_path.read_text()
    match = re.search(r"best bound\s+([\d.e+]+)", content)
    return float(match.group(1)) if match else None


def _parse_instance_metadata(instance_name: str) -> dict:
    parts = instance_name.split("_")
    if len(parts) >= 5:
        return {
            "source": parts[0],
            "num_sequences": int(parts[1]),
            "sequence_length": int(parts[2]),
            "alphabet_size": int(parts[3]),
        }
    return {"source": "unknown", "num_sequences": 0, "sequence_length": 0, "alphabet_size": 0}


def extract_lower_bounds() -> None:
    """Extract lower bounds from Gurobi logs and compute static bounds from instances."""
    # Import Instance from the zenodo repo
    sys.path.insert(0, str(REPO_DIR))
    from classes.instance import Instance

    gurobi_dir = RESULTS_DIR / "gurobi"
    if not gurobi_dir.is_dir():
        logger.error("gurobi results directory not found at %s", gurobi_dir)
        return

    log_files = sorted(
        f for f in gurobi_dir.rglob("stderr.log") if "stderr_raw" not in f.parts
    )
    logger.info("Found %d gurobi stderr.log files for lower bounds", len(log_files))

    results = []
    for i, log_path in enumerate(log_files, 1):
        try:
            content = log_path.read_text()
        except Exception:
            continue

        instance_name = _parse_instance_name_from_log(content)
        if not instance_name:
            continue

        gurobi_bound = _parse_lb_stderr(content)
        if gurobi_bound is None:
            stdout_path = log_path.with_name("stdout.log")
            gurobi_bound = _parse_lb_stdout(stdout_path)
            if gurobi_bound is None:
                continue

        metadata = _parse_instance_metadata(instance_name)

        results.append({
            "Instances": instance_name,
            "Source": metadata["source"],
            "num_sequences": metadata["num_sequences"],
            "sequence_length": metadata["sequence_length"],
            "alphabet_size": metadata["alphabet_size"],
            "gurobi_bound": gurobi_bound,
        })

        if i % 50 == 0:
            logger.info("Parsed %d/%d gurobi logs", i, len(log_files))

    if not results:
        logger.warning("No lower bound results extracted")
        return

    df = pd.DataFrame(results)

    # Compute static bounds from instance files
    static_bounds = []
    for instance_name in df["Instances"].unique():
        instance_path = INSTANCES_DIR / f"{instance_name}.dat"
        if instance_path.exists():
            try:
                inst = Instance.from_file(str(instance_path))
                static_bounds.append({
                    "Instances": instance_name,
                    "static_bound": inst.static_bound,
                })
            except Exception as e:
                logger.warning("Failed to compute static bound for %s: %s", instance_name, e)
        else:
            logger.warning("Instance file not found: %s", instance_path)

    if static_bounds:
        sb_df = pd.DataFrame(static_bounds)
        df = df.merge(sb_df, on="Instances", how="left")
    else:
        df["static_bound"] = np.nan

    df["lower_bound"] = df[["gurobi_bound", "static_bound"]].max(axis=1)
    df.sort_values(
        by=["Source", "alphabet_size", "sequence_length", "num_sequences"], inplace=True
    )

    out_path = OUTPUT_DIR / "lower_bounds.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d lower bound records to %s", len(df), out_path.name)


# ===================================================================
# Step 3 — Merge summaries  (from 2.1-summaries_merger_from_stderr.py)
# ===================================================================


def merge_summaries() -> None:
    """Merge per-algorithm stderr summary CSVs into a single summary_merged.csv."""
    dfs = []
    for dir_name, algo_name in ALGO_DIR_TO_NAME.items():
        csv_path = OUTPUT_DIR / f"{dir_name}_stderr_summary.csv"
        if not csv_path.exists():
            logger.warning("Missing %s — skipping", csv_path.name)
            continue
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            logger.error("Error reading %s: %s", csv_path.name, e)
            continue
        if df.empty:
            continue

        # Normalise algorithm column
        if "algorithm" not in df.columns:
            df["algorithm"] = algo_name

        # Normalise objective column name
        if "objective_value" not in df.columns and "objective_function_value" in df.columns:
            df.rename(columns={"objective_function_value": "objective_value"}, inplace=True)

        if "objective_value" not in df.columns:
            logger.warning("No objective column in %s — skipping", csv_path.name)
            continue

        dfs.append(df)

    if not dfs:
        logger.error("No valid summary files found to merge.")
        return

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # Merge lower bounds
    lb_path = OUTPUT_DIR / "lower_bounds.csv"
    if lb_path.exists():
        try:
            lb_df = pd.read_csv(lb_path, low_memory=False)
            if "Instances" in lb_df.columns:
                lb_col = None
                for candidate in ("lower_bound", "lower_bounds"):
                    if candidate in lb_df.columns:
                        lb_col = candidate
                        break
                if lb_col:
                    if lb_col != "lower_bounds":
                        lb_df = lb_df.rename(columns={lb_col: "lower_bounds"})
                    merged = merged.merge(
                        lb_df[["Instances", "lower_bounds"]], on="Instances", how="left"
                    )
        except Exception as e:
            logger.error("Error reading lower bounds: %s", e)

    # Add Source column
    if "Instances" in merged.columns:
        merged["Source"] = merged["Instances"].apply(
            lambda x: "balanced" if "balanced" in str(x) else "uniform"
        )

    # Add instance metadata columns for plotting
    for col, idx in [("num_sequences", 1), ("sequence_length", 2), ("alphabet_size", 3)]:
        if col not in merged.columns:
            merged[col] = merged["Instances"].apply(
                lambda x, i=idx: int(str(x).split("_")[i]) if len(str(x).split("_")) > i else 0
            )

    out_path = OUTPUT_DIR / "summary_merged.csv"
    merged.to_csv(out_path, index=False)
    logger.info("Saved merged summary with %d rows to %s", len(merged), out_path.name)


# ===================================================================
# Step 4 — Create LaTeX tables  (from 3-table_creator.py)
# ===================================================================


def _parse_instance_meta(instance_name: str) -> dict:
    parts = str(instance_name).split("_")
    return {
        "index": int(parts[-1]),
        "m": int(parts[2]),
        "n": int(parts[1]),
        "k": int(parts[3]),
    }


def _pivot_results(df: pd.DataFrame) -> pd.DataFrame:
    pivoted = df.pivot_table(
        index="Instances",
        columns="algorithm",
        values="objective_value",
        aggfunc="mean",
        dropna=False,
        observed=False,
    )
    pivoted.columns = [f"{col}_mean" for col in pivoted.columns]
    for col in ALGORITHM_COLUMNS:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    metadata = pd.DataFrame(
        [_parse_instance_meta(idx) for idx in pivoted.index], index=pivoted.index
    )
    return pd.concat([metadata, pivoted[ALGORITHM_COLUMNS]], axis=1)


def _format_value(value: float, is_minimum: bool, as_integer: bool = False) -> str:
    if pd.isna(value):
        return ""
    formatted = str(int(value)) if as_integer else f"{value:.1f}"
    return rf"\bfseries {formatted}" if is_minimum else formatted


def _create_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    lower_bounds: Optional[pd.DataFrame] = None,
) -> str:
    df = df.copy().reset_index()
    if "Instances" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "Instances"})

    if lower_bounds is not None:
        df = df.merge(lower_bounds[["Instances", "lower_bound"]], on="Instances", how="left")
    else:
        df["lower_bound"] = np.nan

    df = df.sort_values(by=["k", "m", "n", "index"]).reset_index(drop=True)

    s_col = r"S[table-format=9.1, detect-weight, mode=text]"
    header_row = (
        r"index & $m$ & $n$ & $k$ & LB & "
        + " & ".join(ALGORITHM_DISPLAY_NAMES_TABLE)
        + r" \\"
    )

    lines = [
        r"\begin{longtable}[htp]{*{4}{c}r*{3}{" + s_col + r"}}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endfirsthead",
        rf'\caption[]{{{caption.split(".")[0]}.}} \\',
        r"\toprule",
        header_row,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for _, row in df.iterrows():
        idx = int(row["index"])
        m = int(row["m"])
        n = int(row["n"])
        k = int(row["k"])
        lb = int(row["lower_bound"]) if pd.notna(row["lower_bound"]) else ""

        algo_values = [row[col] for col in ALGORITHM_COLUMNS if pd.notna(row[col])]
        min_val = min(algo_values) if algo_values else None

        values = []
        for col in ALGORITHM_COLUMNS:
            val = row[col]
            if pd.isna(val):
                values.append("")
            else:
                is_gurobi = col == "gurobi_mean"
                formatted = str(int(val)) if is_gurobi else f"{val:.1f}"
                if min_val is not None and val == min_val:
                    formatted = r"\bfseries " + formatted
                values.append(formatted)

        lines.append(f"{idx} & {m} & {n} & {k} & {lb} & " + " & ".join(values) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _create_gap_table(
    pivot_df: pd.DataFrame, lower_bounds: pd.DataFrame, name_suffix: str
) -> None:
    lower_bounds = lower_bounds.drop_duplicates(subset="Instances")
    pivot_reset = pivot_df[ALGORITHM_COLUMNS].reset_index()
    merged = pivot_reset.merge(
        lower_bounds[["Instances", "lower_bound"]], on="Instances", how="left"
    )

    lb = pd.to_numeric(merged["lower_bound"], errors="coerce")
    valid_lb = lb.notna() & (lb != 0)

    gaps = pd.DataFrame()
    for col in ALGORITHM_COLUMNS:
        gap_col = col.replace("_mean", "_gap")
        gaps[gap_col] = np.where(valid_lb, (merged[col] - lb) / lb * 100, np.nan)

    gap_columns = [col.replace("_mean", "_gap") for col in ALGORITHM_COLUMNS]
    row_min = gaps[gap_columns].min(axis=1)

    formatted_gaps = {}
    for col in gap_columns:
        is_min = gaps[col].eq(row_min)
        formatted_gaps[col] = [
            _format_value(v, bold) for v, bold in zip(gaps[col], is_min)
        ]

    out_df = pd.DataFrame({
        "index": pivot_df["index"].astype(int).values,
        "m": pivot_df["m"].astype(int).values,
        "n": pivot_df["n"].astype(int).values,
        "k": pivot_df["k"].astype(int).values,
        **formatted_gaps,
    })

    gap_header = ["index", r"$m$", "$n$", "$k$", "Gurobi gap", "SA gap", "HG2 gap"]
    table_str = out_df.to_latex(
        header=gap_header,
        columns=["index", "m", "n", "k"] + gap_columns,
        index=False,
        escape=False,
        longtable=True,
    )

    output_path = OUTPUT_DIR / f"summary_merged_gaps_{name_suffix}.tex"
    output_path.write_text(table_str)
    logger.info("Saved gap table to %s", output_path.name)


def _create_average_gap_table(
    df: pd.DataFrame, lower_bounds: pd.DataFrame, name_suffix: str
) -> None:
    df = df[df["algorithm"] != "gain_heuristic"].copy()
    lower_bounds = lower_bounds.drop_duplicates(subset="Instances")

    df = df.merge(lower_bounds[["Instances", "lower_bound"]], on="Instances", how="left")
    df["gap"] = (df["objective_value"] - df["lower_bound"]) / df["lower_bound"] * 100

    avg_gaps = df.groupby(["algorithm", "Instances"])["gap"].mean().reset_index()
    pivoted = avg_gaps.pivot(index="Instances", columns="algorithm", values="gap").reset_index()

    meta = pivoted["Instances"].apply(lambda x: pd.Series(_parse_instance_meta(x)))
    pivoted = pd.concat([pivoted, meta[["m", "n", "k"]].reset_index(drop=True)], axis=1)

    group_keys = ["n", "m", "k"]
    alg_cols = [c for c in GAP_COLUMN_ORDER if c in pivoted.columns]
    grouped = pivoted.groupby(group_keys)[alg_cols].mean().reset_index()
    grouped = grouped.sort_values(by=group_keys).reset_index(drop=True)

    for col in alg_cols:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    scaled = grouped[alg_cols] * GAP_SCALE
    row_min = scaled.min(axis=1)

    formatted = {}
    for col in alg_cols:
        is_min = scaled[col].eq(row_min)
        formatted[col] = [_format_value(v, bold) for v, bold in zip(scaled[col], is_min)]

    n_rows = len(grouped)
    half = (n_rows + 1) // 2

    s_col = r"S[table-format=4.1, detect-weight, mode=text]"

    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{Average GAPs (scaled by $10^3$) for {name_suffix} instances.}}",
        rf"\label{{tab:avg_gaps_{name_suffix}}}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{rrr*{{3}}{{{s_col}}} c rrr*{{3}}{{{s_col}}}}}",
        r"\toprule",
        r"{$n$} & {$m$} & {$\Sigma$} & {SA} & {Gurobi} & {HG2} & & {$n$} & {$m$} & {$\Sigma$} & {SA} & {Gurobi} & {HG2} \\",
        r"\midrule",
    ]

    for i in range(half):
        left_idx = i
        right_idx = i + half

        n_left = int(grouped.iloc[left_idx]["n"])
        m_left = int(grouped.iloc[left_idx]["m"])
        k_left = int(grouped.iloc[left_idx]["k"])
        sa_left = formatted["simulated_annealing"][left_idx]
        grb_left = formatted["gurobi"][left_idx]
        hg2_left = formatted["hg2"][left_idx]
        left_part = f"{n_left} & {m_left} & {k_left} & {sa_left} & {grb_left} & {hg2_left}"

        if right_idx < n_rows:
            n_right = int(grouped.iloc[right_idx]["n"])
            m_right = int(grouped.iloc[right_idx]["m"])
            k_right = int(grouped.iloc[right_idx]["k"])
            sa_right = formatted["simulated_annealing"][right_idx]
            grb_right = formatted["gurobi"][right_idx]
            hg2_right = formatted["hg2"][right_idx]
            right_part = f"{n_right} & {m_right} & {k_right} & {sa_right} & {grb_right} & {hg2_right}"
        else:
            right_part = "& & & & &"

        lines.append(f"{left_part} && {right_part} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    output_path = OUTPUT_DIR / f"average_gaps_{name_suffix}.tex"
    output_path.write_text("\n".join(lines))
    logger.info("Saved average gap table to %s", output_path.name)


def create_tables() -> None:
    """Create all pivoted CSVs and LaTeX tables."""
    df = pd.read_csv(OUTPUT_DIR / "summary_merged.csv")
    df = df.drop_duplicates()

    df_balanced = df[df["Source"] == "balanced"]
    df_uniform = df[df["Source"] == "uniform"]

    pivoted_balanced = _pivot_results(df_balanced)
    pivoted_uniform = _pivot_results(df_uniform)

    pivoted_balanced.to_csv(OUTPUT_DIR / "pivoted_balanced.csv")
    pivoted_uniform.to_csv(OUTPUT_DIR / "pivoted_uniform.csv")

    pivoted_balanced["Source"] = "balanced"
    pivoted_uniform["Source"] = "uniform"
    pivoted_all = pd.concat([pivoted_balanced, pivoted_uniform], ignore_index=True)
    pivoted_all.to_csv(OUTPUT_DIR / "pivoted_all.csv")
    logger.info("Saved pivoted dataframes")

    # Count SA-better instances
    def count_sa_better(pivot_df: pd.DataFrame) -> tuple[int, int]:
        sa = pivot_df.get("simulated_annealing_mean")
        grb = pivot_df.get("gurobi_mean")
        hg2 = pivot_df.get("hg2_mean")
        better_vs_grb = int(((sa.notna()) & (grb.notna()) & (sa < grb)).sum())
        better_vs_hg2 = int(((sa.notna()) & (hg2.notna()) & (sa < hg2)).sum())
        return better_vs_grb, better_vs_hg2

    bal_vs_grb, bal_vs_hg2 = count_sa_better(pivoted_balanced)
    uni_vs_grb, uni_vs_hg2 = count_sa_better(pivoted_uniform)

    counts_path = OUTPUT_DIR / "sa_better_counts.txt"
    counts_path.write_text(
        "\n".join([
            f"balanced,sa_vs_gurobi,{bal_vs_grb}",
            f"balanced,sa_vs_hg2,{bal_vs_hg2}",
            f"uniform,sa_vs_gurobi,{uni_vs_grb}",
            f"uniform,sa_vs_hg2,{uni_vs_hg2}",
        ])
        + "\n"
    )
    logger.info("Wrote SA better counts")

    lb_path = OUTPUT_DIR / "lower_bounds.csv"
    lb = pd.read_csv(lb_path) if lb_path.exists() else None

    table_balanced = _create_latex_table(
        pivoted_balanced,
        "Aggregated main results for balanced instances. SA and HG2 are averaged over 10 runs. The best performing method is highlighted in bold.",
        "tab:results_balanced",
        lower_bounds=lb,
    )
    (OUTPUT_DIR / "summary_merged_grouped_balanced.tex").write_text(table_balanced)

    table_uniform = _create_latex_table(
        pivoted_uniform,
        "Aggregated main results for uniform instances. SA and HG2 are averaged over 10 runs. The best performing method is highlighted in bold.",
        "tab:results_uniform",
        lower_bounds=lb,
    )
    (OUTPUT_DIR / "summary_merged_grouped_uniform.tex").write_text(table_uniform)
    logger.info("Saved balanced/uniform results tables")

    if lb is not None:
        _create_gap_table(pivoted_balanced, lb, "balanced")
        _create_gap_table(pivoted_uniform, lb, "uniform")
        _create_average_gap_table(df_balanced, lb, "balanced")
        _create_average_gap_table(df_uniform, lb, "uniform")


# ===================================================================
# Step 5 — Create plots  (from 4-plotter.py)
# ===================================================================


def _load_plot_data() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "summary_merged.csv")
    df = df[df["algorithm"] != "gain_heuristic"]
    df["algorithm"] = df["algorithm"].apply(_rename_algorithm)
    df["index"] = df["Instances"].str.split("_").str[-1]

    lb = pd.read_csv(OUTPUT_DIR / "lower_bounds.csv")
    lb = lb.drop_duplicates(subset="Instances")
    lb_series = lb.set_index("Instances")["lower_bound"]
    df["lower_bound"] = df["Instances"].map(lb_series)
    df["GAP"] = (df["objective_value"] - df["lower_bound"]) / df["lower_bound"] * 100

    return df


def _plot_main_results(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = _get_algo_colors()

    df_balanced = df[df["Source"] == "balanced"]
    df_uniform = df[df["Source"] == "uniform"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Violin by instance type
    ax = axes[0]
    data_for_violin = []
    for source, df_src in [("Balanced", df_balanced), ("Uniform", df_uniform)]:
        for algo in ["GRB", "SA", "HG2"]:
            df_algo = df_src[df_src["algorithm"] == algo]
            for val in df_algo["GAP"].values:
                data_for_violin.append({"Instance Type": source, "Algorithm": algo, "Gap (%)": val})
    df_violin = pd.DataFrame(data_for_violin)

    sns.violinplot(
        data=df_violin, x="Instance Type", y="Gap (%)", hue="Algorithm",
        ax=ax, palette=colors, inner="quartile", cut=0, linewidth=1.5,
    )
    ax.set_ylabel("Gap from Lower Bound (%)", fontsize=13)
    ax.set_xlabel("Instance Type", fontsize=13)
    ax.set_title("(A) Algorithm Performance by Instance Type", fontsize=14, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, title="Algorithm", title_fontsize=11, loc="upper right")
    ax.set_ylim(bottom=0)

    # Panel B: Scalability by n
    ax = axes[1]
    bal_by_n = df_balanced.groupby(["num_sequences", "algorithm"])["GAP"].agg(["mean", "std"]).reset_index()

    for algo in ["GRB", "SA", "HG2"]:
        df_algo = bal_by_n[bal_by_n["algorithm"] == algo]
        ax.errorbar(
            df_algo["num_sequences"], df_algo["mean"], yerr=df_algo["std"],
            label=algo, color=colors[algo], marker="o", markersize=8,
            linewidth=2, capsize=5, capthick=2,
        )

    ax.set_xlabel("Number of Sequences (n)", fontsize=13)
    ax.set_ylabel("Mean Gap (%)", fontsize=13)
    ax.set_title("(B) Scalability with Instance Size", fontsize=14, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    # Panel C: Alphabet size effect
    ax = axes[2]
    bal_by_k = df_balanced.groupby(["alphabet_size", "algorithm"])["GAP"].agg(["mean", "std"]).reset_index()

    x_pos = np.arange(len(bal_by_k["alphabet_size"].unique()))
    width_bar = 0.25

    for i, algo in enumerate(["GRB", "SA", "HG2"]):
        df_algo = bal_by_k[bal_by_k["algorithm"] == algo]
        ax.bar(
            x_pos + i * width_bar, df_algo["mean"], width_bar,
            yerr=df_algo["std"], label=algo, color=colors[algo], capsize=5,
        )

    ax.set_xlabel("Alphabet Size (k)", fontsize=13)
    ax.set_ylabel("Mean Gap (%)", fontsize=13)
    ax.set_title("(C) Effect of Alphabet Size", fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(x_pos + width_bar)
    ax.set_xticklabels(sorted(df_balanced["alphabet_size"].unique()))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "figure_main_results.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / "figure_main_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Created figure_main_results.pdf/.png")


def _plot_detailed_comparison(df: pd.DataFrame, source_name: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = _get_algo_colors()
    df_source = df[df["Source"] == source_name]

    m_values = sorted(df_source["sequence_length"].unique())
    k_values = sorted(df_source["alphabet_size"].unique())

    fig, axes = plt.subplots(
        len(k_values), len(m_values),
        figsize=(4 * len(m_values), 3.5 * len(k_values)),
        sharex=True, sharey=True, squeeze=False,
    )

    for i, k in enumerate(k_values):
        for j, m in enumerate(m_values):
            ax = axes[i, j]
            df_subset = df_source[
                (df_source["sequence_length"] == m) & (df_source["alphabet_size"] == k)
            ]

            sns.boxplot(
                data=df_subset, x="num_sequences", y="GAP",
                hue="algorithm", ax=ax, palette=colors, linewidth=1.5, showfliers=False,
            )

            ax.set_title(f"m={m}, k={k}", fontsize=12, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")

            if i == len(k_values) - 1:
                ax.set_xlabel("n", fontsize=11)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel("Gap (%)", fontsize=11)
            else:
                ax.set_ylabel("")
            if i == 0 and j == len(m_values) - 1:
                ax.legend(fontsize=10, title="Algorithm", title_fontsize=10)
            elif ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()
    fname = f"figure_detailed_{source_name}"
    plt.savefig(PLOTS_DIR / f"{fname}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(PLOTS_DIR / f"{fname}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Created %s.pdf/.png", fname)


def _plot_boxplot_by_instance(df: pd.DataFrame, source_name: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    colors = _get_algo_colors()
    fontsize = 18
    df_source = df[df["Source"] == source_name]

    for alphabet_size in df_source["alphabet_size"].unique():
        df_k = df_source[df_source["alphabet_size"] == alphabet_size]
        for seq_length in df_k["sequence_length"].unique():
            df_m = df_k[df_k["sequence_length"] == seq_length]
            for num_seq in df_m["num_sequences"].unique():
                df_subset = df_m[df_m["num_sequences"] == num_seq]

                fig, ax = plt.subplots()
                sns.boxplot(
                    data=df_subset, x="index", y="GAP", hue="algorithm",
                    ax=ax, palette=colors, legend=False,
                )
                ax.grid(True, axis="y", alpha=0.3, linestyle="--")
                ax.tick_params(labelsize=fontsize)
                ax.set_xlabel("Instance Index", fontsize=fontsize)
                ax.set_ylabel("GAP (%)", fontsize=fontsize)
                ax.set_title(
                    f"n={num_seq}, m={seq_length}, k={alphabet_size}",
                    fontsize=fontsize, fontweight="bold",
                )

                filename = f"boxplot_{source_name}_{num_seq}_{seq_length}_{alphabet_size}"
                plt.savefig(PLOTS_DIR / f"{filename}.pdf", bbox_inches="tight")
                plt.savefig(PLOTS_DIR / f"{filename}.png", bbox_inches="tight")
                plt.close(fig)


def create_plots() -> None:
    """Generate all paper figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("tableau-colorblind10")
    sns.set_palette("colorblind")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_plot_data()
    _plot_main_results(df)

    for source in ["balanced", "uniform"]:
        _plot_detailed_comparison(df, source)
        _plot_boxplot_by_instance(df, source)

    logger.info("All plots saved to %s", PLOTS_DIR)


# ===================================================================
# Step 6 — Win/loss/draw statistics  (from 5-pie_plotter.py)
# ===================================================================


def _compute_win_loss_draw(df: pd.DataFrame, algorithms: list[str]) -> pd.DataFrame:
    results = {algo: {"wins": 0, "losses": 0, "draws": 0} for algo in algorithms}

    for _, row in df.iterrows():
        values = {algo: row[algo] for algo in algorithms}
        min_value = min(values.values())
        winners = [algo for algo, val in values.items() if val == min_value]

        if len(winners) == 1:
            results[winners[0]]["wins"] += 1
            for algo in algorithms:
                if algo != winners[0]:
                    results[algo]["losses"] += 1
        else:
            for algo in winners:
                results[algo]["draws"] += 1
            for algo in algorithms:
                if algo not in winners:
                    results[algo]["losses"] += 1

    return pd.DataFrame(results).T.reset_index().rename(columns={"index": "algorithm"})


def compute_comparisons() -> None:
    """Compute win/loss/draw statistics from pivoted CSVs."""
    for name in ["pivoted_balanced", "pivoted_uniform"]:
        csv_path = OUTPUT_DIR / f"{name}.csv"
        if not csv_path.exists():
            logger.warning("Skipping missing %s", csv_path.name)
            continue

        df = pd.read_csv(csv_path)
        wld = _compute_win_loss_draw(df, ALGORITHM_COLUMNS)

        logger.info("%s:\n%s", name, wld.to_string(index=False))

        output_path = OUTPUT_DIR / f"algorithm_comparison_{name}.tex"
        wld.to_latex(output_path, index=False)
        logger.info("Saved %s", output_path.name)


# ===================================================================
# Step 7 — Gap statistics  (from 6-gap_calculator.py)
# ===================================================================


def _compute_gaps(df: pd.DataFrame, lb: pd.DataFrame) -> pd.DataFrame:
    df = df[df["algorithm"] != "gain_heuristic"].copy()
    df["instance_index"] = df["Instances"].apply(lambda x: int(x.split("_")[-1]))
    df = df.merge(lb[["Instances", "lower_bound"]], on="Instances", how="left")
    df["gap"] = (df["objective_value"] - df["lower_bound"]) / df["lower_bound"] * 100
    return df


def _pivot_gaps(df: pd.DataFrame) -> pd.DataFrame:
    instance_info = df[["Instances", "instance_index", "Source"]].drop_duplicates()
    average_gaps = df.groupby(["algorithm", "Instances"])["gap"].mean().reset_index()
    pivoted = average_gaps.pivot(index="Instances", columns="algorithm", values="gap").reset_index()
    return pivoted.merge(instance_info, on="Instances", how="left")


def _compute_mean_gaps(pivoted: pd.DataFrame) -> pd.DataFrame:
    alg_cols = [c for c in GAP_COLUMN_ORDER if c in pivoted.columns]
    return pivoted.groupby("Source")[alg_cols].mean(numeric_only=True).reset_index()


def _create_gap_latex_table_simple(df_group: pd.DataFrame, source_name: str) -> None:
    alg_cols = [c for c in GAP_COLUMN_ORDER if c in df_group.columns]

    for col in alg_cols:
        df_group[col] = pd.to_numeric(df_group[col], errors="coerce")

    scaled = df_group[alg_cols] * GAP_SCALE
    row_min = scaled.min(axis=1)

    formatted = {}
    for col in alg_cols:
        is_min = scaled[col].eq(row_min)
        formatted[col] = [_format_value(v, bold) for v, bold in zip(scaled[col], is_min)]

    out_df = pd.DataFrame({
        GAP_DISPLAY_NAMES.get(col, col): formatted[col] for col in alg_cols
    })

    table_str = out_df.to_latex(
        index=False, escape=False, longtable=False,
        caption=f"Average gaps (scaled by $10^3$) for {source_name} instances.",
        label=f"tab:mean_gaps_{source_name}",
    )

    output_path = OUTPUT_DIR / f"mean_gaps_{source_name}.tex"
    output_path.write_text(table_str)
    logger.info("Saved %s", output_path.name)


def calculate_gaps() -> None:
    """Calculate and export gap statistics."""
    df = pd.read_csv(OUTPUT_DIR / "summary_merged.csv")
    lb = pd.read_csv(OUTPUT_DIR / "lower_bounds.csv")

    df_with_gaps = _compute_gaps(df, lb)
    pivoted = _pivot_gaps(df_with_gaps)
    pivoted.to_csv(OUTPUT_DIR / "pivoted_gaps.csv", index=False)

    grouped = _compute_mean_gaps(pivoted)
    grouped.to_csv(OUTPUT_DIR / "mean_gaps.csv", index=False)

    balanced = grouped[grouped["Source"] == "balanced"].copy()
    uniform = grouped[grouped["Source"] == "uniform"].copy()

    _create_gap_latex_table_simple(balanced, "balanced")
    _create_gap_latex_table_simple(uniform, "uniform")


# ===================================================================
# Step 8 — Generate CSVs for statistical evaluation (R script)
# ===================================================================

STAT_EVAL_DIR = REPO_DIR / "statistical_evaluation"

STAT_COLUMN_MAP = {
    "simulated_annealing_mean": "SA",
    "gurobi_mean": "Gurobi",
    "hg2_mean": "HG2",
}


def _pivoted_to_stat_csv(pivoted_csv: Path, output_csv: Path) -> None:
    """Convert a pivoted results CSV to the format expected by statistical_evaluation.R.

    Mapping: index → inst, simulated_annealing_mean → SA, gurobi_mean → Gurobi,
    hg2_mean → HG2. All other columns are dropped.
    """
    df = pd.read_csv(pivoted_csv)
    out = df[["index"]].copy()
    out.rename(columns={"index": "inst"}, inplace=True)
    for src, dst in STAT_COLUMN_MAP.items():
        out[dst] = df[src] if src in df.columns else np.nan
    out.to_csv(output_csv, index=False)
    logger.info("Saved statistical evaluation CSV to %s", output_csv)


def generate_statistical_csvs() -> None:
    """Generate results_balanced.csv (and results_uniform.csv) for the R script."""
    STAT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for source in ["balanced", "uniform"]:
        pivoted_path = OUTPUT_DIR / f"pivoted_{source}.csv"
        if not pivoted_path.exists():
            logger.warning("Missing %s — skipping statistical CSV", pivoted_path.name)
            continue
        output_path = STAT_EVAL_DIR / f"results_{source}.csv"
        _pivoted_to_stat_csv(pivoted_path, output_path)


# ===================================================================
# Main
# ===================================================================

STEPS = [
    ("Extract results",       extract_results),            # 0
    ("Parse stderr logs",     parse_stderr_logs),           # 1
    ("Extract lower bounds",  extract_lower_bounds),        # 2
    ("Merge summaries",       merge_summaries),             # 3
    ("Create tables",         create_tables),               # 4
    ("Create plots",          create_plots),                # 5
    ("Compute comparisons",   compute_comparisons),         # 6
    ("Calculate gaps",        calculate_gaps),              # 7
    ("Generate stat CSVs",    generate_statistical_csvs),   # 8
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-processing pipeline for 2-OMF experimental results."
    )
    parser.add_argument(
        "--from-step", type=int, default=0, metavar="N",
        help="Start from step N (0–8). Useful to skip extraction/parsing when CSVs exist.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (name, func) in enumerate(STEPS):
        if i < args.from_step:
            continue
        logger.info("=== Step %d: %s ===", i, name)
        func()

    logger.info("Pipeline complete. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
