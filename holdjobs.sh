#!/bin/bash

# for i in $(seq 106428 106509); do scontrol hold $i; done
# for i in $(seq 106428 106509); do scontrol release $i; done
# for i in $(seq 108445 108508); do scancel $i; done
# squeue -u $USER | grep 197 | awk '{print $1}' | xargs -n 1 scancel