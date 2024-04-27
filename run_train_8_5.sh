#!/bin/bash

set -e

python ./train.py -e 300 -b 64 -s 3 -c 10 --overwrite_sample_number false -m ./layers_8_kernel_5_no_flat.yaml -f /inputs/fair-face-volume/fairface