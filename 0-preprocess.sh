#!/usr/bin/env bash
# =============================================================================
# 0-preprocess.sh — Extract benchmark instances and experiment results
# =============================================================================
#
# Run this once before executing experiments or post-processing.
#
# Usage:
#   bash 0-preprocess.sh
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Instances ----
INST_ARCHIVE="$SCRIPT_DIR/data/instances.tar.gz"
INST_TARGET="$SCRIPT_DIR/instances"

if [ -d "$INST_TARGET" ] && [ "$(ls -A "$INST_TARGET"/*.dat 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "instances/ already contains .dat files. Skipping extraction."
else
    if [ ! -f "$INST_ARCHIVE" ]; then
        echo "ERROR: $INST_ARCHIVE not found."
        exit 1
    fi
    echo "Extracting benchmark instances..."
    tar xzf "$INST_ARCHIVE" -C "$SCRIPT_DIR"
    COUNT=$(ls "$INST_TARGET"/*.dat 2>/dev/null | wc -l)
    echo "Done. Extracted $COUNT instance files to instances/"
fi

# ---- Results (experiment logs) ----
RES_ARCHIVE="$SCRIPT_DIR/data/results.tar.gz"
RES_TARGET="$SCRIPT_DIR/results"

if [ -d "$RES_TARGET" ] && [ "$(find "$RES_TARGET" -name 'stderr.log' 2>/dev/null | head -1)" ]; then
    echo "results/ already contains log files. Skipping extraction."
else
    if [ ! -f "$RES_ARCHIVE" ]; then
        echo "NOTE: $RES_ARCHIVE not found. Skipping results extraction."
        echo "      (Only needed for post-processing: python 2-postprocessing.py)"
    else
        echo "Extracting experiment results..."
        tar xzf "$RES_ARCHIVE" -C "$SCRIPT_DIR"
        COUNT=$(find "$RES_TARGET" -name 'stderr.log' 2>/dev/null | wc -l)
        echo "Done. Extracted $COUNT log files to results/"
    fi
fi
