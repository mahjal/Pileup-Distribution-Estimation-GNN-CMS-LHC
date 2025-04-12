# Pileup Distribution Estimation using GNNs (CMS LHC)
This project leverages Graph Neural Networks (GAT, GraphSAGE) for estimating pileup distributions at the CMS detector of the LHC. The model processes particle collision data to predict pileup conditions in high-energy physics experiments.

# 📁 Project Structure

    src/ — Contains training, evaluation, and model-related scripts.

    utils/ — Includes data loaders, metrics, and utility functions to support training and evaluation.

    data/ — Stores the homogeneous data from /eos/cms/store/user/hbakhshi/PUGNN/.

    reports/ — Contains thesis drafts, short reports, and research documentation.

    notebooks/ — Jupyter notebooks providing demos and data visualizations.

# 🛠️ Setup
## 1. Install dependencies

Make sure Python 3.9 or higher is installed. Then, install the project dependencies by running:

pip install -r requirements.txt

## 2. Virtual Environment (Optional but recommended)

You can set up a virtual environment to isolate project dependencies:

python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt

## 3. Python Environment Configuration

Ensure your environment is correctly configured to run the project. Set the PYTHONPATH to include the src directory:

export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/user/m/mjalalva/Pileup-Distribution-Estimation-GNN-CMS-LHC/src




