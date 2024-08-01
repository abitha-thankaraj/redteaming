#!/bin/bash

# Define directories
DIR1="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/harmbench_chunked/"
DIR2="/data/group_data/rl/datasets/redteaming/gen_multiturn_prompts/openai_chunked/"

# Define the pattern to search for within the file names
PATTERN="*.json"

# Find all files that match the pattern in both directories and store them in a variable
FILES=$(find $DIR1 $DIR2 -type f -name "$PATTERN")

# Use the variable
echo "Found files:"
echo "$FILES"
NUM_FILES=$(echo "$FILES" | grep -c '^')
echo "Number of files: $NUM_FILES"
