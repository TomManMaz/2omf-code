#!/usr/bin/env bash
# =============================================================================
# 1-preprocess.sh — Extract benchmark instances
# =============================================================================
#
# Run this once before executing any experiments.
#
# Usage:
#   bash 1-preprocess.sh
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="$SCRIPT_DIR/instances.tar.gz"
TARGET="$SCRIPT_DIR/instances"

if [ -d "$TARGET" ] && [ "$(ls -A "$TARGET"/*.dat 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "instances/ already contains .dat files. Skipping extraction."
    echo "To re-extract, remove the directory first: rm -rf $TARGET"
    exit 0
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: $ARCHIVE not found."
    exit 1
fi

echo "Extracting benchmark instances..."
tar xzf "$ARCHIVE" -C "$SCRIPT_DIR"

COUNT=$(ls "$TARGET"/*.dat 2>/dev/null | wc -l)
echo "Done. Extracted $COUNT instance files to instances/"
