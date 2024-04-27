#!/bin/bash

set -e

python ./train.py -e 4 -b 64 -s 3 -c 1 --overwrite_sample_number true --number_samples 100 -m ./layers_8_kernel_3_no_flat.yaml -f /inputs/fair-face-volume/fairface