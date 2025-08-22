#!/bin/bash

# Create conda environment
conda create -n trade python=3.11 -y

# Wait 5 seconds
sleep 5

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate trade

# Install requirements
python -m pip install -r requirements.txt
