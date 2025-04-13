#!/bin/bash

# Detect OS version
OS=$(uname -r)

if [[ $OS == *"el7"* ]]; then
    echo "Using CentOS 7"
    source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-centos7-gcc11-opt/setup.sh
elif [[ $OS == *"el8"* ]]; then
    echo "Using CentOS 8"
    source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-centos8-gcc11-opt/setup.sh
elif [[ $OS == *"el9"* ]]; then
    echo "Using CentOS 9"
    source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh
else
    echo "Unknown OS: $OS"
fi

# Setup virtual environment
python -m virtualenv pu-gnn
source pu-gnn/bin/activate

# Load necessary modules
#module load python/3.8
#module load cuda/11.2

# Set the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/user/m/mjalalva/Pileup-Distribution-Estimation-GNN-CMS-LHC/src

cp -r ~/Pileup-Distribution-Estimation-GNN-CMS-LHC/src .

# Run training
python condor_jobs/train_gat.py
