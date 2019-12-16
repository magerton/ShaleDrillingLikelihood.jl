#!/bin/bash
echo $JULIA_NUM_THREADS
ORIGTHREADS=$JULIA_NUM_THREADS
for i in 1 2 4 8 16 32 64
do
    export JULIA_NUM_THREADS=$i
    julia "./full-model/optimize-dynamic-speedtest.jl"
done
export JULIA_NUM_THREADS=$ORIGTHREADS
