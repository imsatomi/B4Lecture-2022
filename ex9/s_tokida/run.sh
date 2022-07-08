#!/bin/bash
. /home/s_tokida/workspace5/venv/etc/profile.d/conda.sh
conda activate ex9

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
 
python main.py
