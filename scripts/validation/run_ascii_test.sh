#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 scripts/verify_ascii_bottleneck.py > ascii_test_results.txt 2>&1
