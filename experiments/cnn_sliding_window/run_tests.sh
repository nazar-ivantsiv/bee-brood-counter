#!/bin/bash

# Activate virtual environment
source ./bee_brood_counter/bin/activate

# Set threading environment variables to avoid TensorFlow issues
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# Run the test
python "$@"
