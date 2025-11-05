#!/bin/bash

set -euo pipefail

python3 -m scripts.download_eval_events "$@"

mkdir -p datasets/train_multiscene datasets/train_singlescene
printf '%s\n' "Evaluation event archives are extracted under ./datasets."
printf '%s\n' "Prepare WorldFloods training data separately via ml4floods and place it under ./datasets/train_multiscene or ./datasets/train_singlescene."
