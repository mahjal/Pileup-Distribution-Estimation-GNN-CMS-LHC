#!/bin/bash

uname -a
OS='uname -r'

if [[ $OS==*"el7"* ]]; then
        echo slc7
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-centos7-gcc11-opt/setup.sh
elif [[ $OS==*"el8"* ]]; then
        echo slc8
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-centos8-gcc11-opt/setup.sh
elif [[ $OS==*"el9"* ]]; then
        echo slc9
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104cuda/x86_64-el9-gcc11-opt/setup.sh
else
        echo $OS
        echo "Unkown OS"
fi

python -m virtualenv pu-gnn
source pu-gnn/bin/activate
pip install -U kaleido
#cp -r /eos/user/c/cmstandi/SWAN_projects/saleh-highlights/PUGNN/PUGNN ./

cp -r /eos/user/m/mjalalva/SWAN_projects/inConc/PUGNN/PUGNN ./


python python_script.py
