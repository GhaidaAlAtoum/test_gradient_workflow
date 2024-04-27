#!/bin/bash

set -e

# --overwrite_sample_number true --number_samples 1000

python ./train.py -e 300 -b 64 -s 3 -c 10  -m ./layers_8_kernel_3_no_flat.yaml -f /inputs/fair-face-volume/fairface