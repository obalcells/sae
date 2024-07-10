#!/bin/bash

# Default values
DATASET="togethercomputer/RedPajama-Data-1T-Sample"
NUM_SAMPLES=10000
N_BATCHES=128
LAYERS=12
BACKGROUND=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --n_batches)
      N_BATCHES="$2"
      shift 2
      ;;
    --background)
      BACKGROUND=true
      shift 1
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if required variables are set
if [ -z "$LAYERS" ]; then
  echo "Error: --layers argument is required"
  exit 1
fi

# Check if the current directory is /root/sae/
if [ "$(pwd)" != "$HOME/sae" ]; then
    echo "Error: Current directory is not $HOME/sae/"
    echo "Please run this script from the $HOME/sae/ directory"
    exit 1
fi

# Construct the command
CMD="python train.py --model meta-llama/Meta-Llama-3-8B --dataset $DATASET --num_dataset_samples $NUM_SAMPLES --load_from_hub_path EleutherAI/sae-llama-3-8b-32x --layers $LAYERS --n_batches $N_BATCHES --run_name base_eleuther_sae"

# Check if we should run in the background
if [[ "$BACKGROUND" == "true" ]]; then
    echo "Running command '$CMD' in the background..."
    nohup $CMD > train_eleuther_base_sae.out 2>&1 &
else
    echo "Running command '$CMD'..."
    $CMD
fi