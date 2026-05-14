#!/bin/bash

# Script to check which datasets have failed sequences after sanity checks
# This reads the output files from check_complete_aws.py

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Directory containing the check results
CHECKS_DIR="$SCRIPT_DIR/../checks"

if [ ! -d "$CHECKS_DIR" ]; then
    echo "Error: Checks directory not found at $CHECKS_DIR"
    echo "Have you run the sanity checks yet?"
    exit 1
fi

echo "========================================"
echo "Dataset Sanity Check Failure Summary"
echo "========================================"
echo ""

total_datasets=0
datasets_with_failures=0
total_failed_sequences=0

# Arrays to store results for sorting
declare -a failed_datasets
declare -a failed_counts

for failed_file in "$CHECKS_DIR"/*_failed.txt; do
    if [ -f "$failed_file" ]; then
        dataset_name=$(basename "$failed_file" _failed.txt)
        total_datasets=$((total_datasets + 1))
        
        # Count non-empty lines (failed sequences)
        fail_count=$(grep -c . "$failed_file" 2>/dev/null || echo 0)
        
        if [ "$fail_count" -gt 0 ]; then
            datasets_with_failures=$((datasets_with_failures + 1))
            total_failed_sequences=$((total_failed_sequences + fail_count))
            failed_datasets+=("$dataset_name")
            failed_counts+=("$fail_count")
        fi
    fi
done

if [ "$total_datasets" -eq 0 ]; then
    echo "No check results found in $CHECKS_DIR"
    echo "Make sure the sanity check jobs have completed."
    exit 0
fi

# Print datasets with failures
if [ "$datasets_with_failures" -gt 0 ]; then
    echo "Datasets with failed sequences:"
    echo "--------------------------------"
    
    # Sort by failure count (descending)
    for i in $(seq 0 $((${#failed_datasets[@]} - 1))); do
        echo "${failed_counts[$i]} ${failed_datasets[$i]}"
    done | sort -rn | while read count name; do
        printf "  %-40s %d failed\n" "$name" "$count"
    done
    
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo "Total datasets checked:      $total_datasets"
    echo "Datasets with failures:      $datasets_with_failures"
    echo "Datasets without failures:   $((total_datasets - datasets_with_failures))"
    echo "Total failed sequences:      $total_failed_sequences"
else
    echo "✓ All $total_datasets datasets passed sanity checks!"
fi

echo ""
echo "----------------------------------------"
echo "Detailed failure logs: $CHECKS_DIR"
echo "----------------------------------------"

