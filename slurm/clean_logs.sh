#!/bin/bash
# Remove SLURM .err and .out log files from the repo root

count=$(ls *.err *.out 2>/dev/null | wc -l)

if [ "$count" -eq 0 ]; then
    echo "No .err or .out files to remove."
    exit 0
fi

echo "Found $count file(s):"
ls *.err *.out 2>/dev/null

read -rp "Delete all? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -f *.err *.out
    echo "Removed $count file(s)."
else
    echo "Aborted."
fi
