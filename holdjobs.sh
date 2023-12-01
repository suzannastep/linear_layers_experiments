#!/bin/bash

# for i in $(seq 105193 105235); do scontrol hold $i; done

for i in $(seq 105193 105235); do scontrol release $i; done