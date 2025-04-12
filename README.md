Pileup Distribution Estimation using GNNs (CMS LHC)

This project leverages Graph Neural Networks (GAT, GraphSAGE) for estimating pileup distributions at the CMS detector of the LHC. The model processes particle collision data to predict pileup conditions in high-energy physics experiments.
📁 Project Structure

    src/ — Contains training, evaluation, and model-related scripts.

    utils/ — Includes data loaders, metrics, and utility functions to support training and evaluation.

    data/ — Stores the homogeneous data from /eos/cms/store/user/hbakhshi/PUGNN/.

    reports/ — Contains thesis drafts, short reports, and research documentation.

    notebooks/ — Jupyter notebooks providing demos and data visualizations.

🛠️ Setup
1. Install dependencies

Make sure Python 3.9 or higher is installed. Then, install the project dependencies by running:

pip install -r requirements.txt

2. Virtual Environment (Optional but recommended)

You can set up a virtual environment to isolate project dependencies:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

3. Python Environment Configuration

Ensure your environment is correctly configured to run the project. Set the PYTHONPATH to include the src directory:

export PYTHONPATH=$PYTHONPATH:/path/to/project/src

For example, you might use:

export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/user/m/mjalalva/Pileup-Distribution-Estimation-GNN-CMS-LHC/src

4. Additional Configuration

If you are using a computing cluster (e.g., CERN), ensure the following modules are loaded:

module load python/3.8
module load cuda/11.2

You may need to adapt paths based on your environment, especially if working in a shared or restricted space like /cvmfs/.
🔧 Running the Code

After setting up the environment, you can run the training scripts located in the condor_jobs/ directory. For example:

cd condor_jobs
bash setup.sh
python train_gat.py

This will initiate the training using GATv2 and your dataset.
📊 Evaluation

Once training is complete, evaluate the model by running:

python evaluate.py

⚡ Notes

    Ensure sufficient disk space is available on the computing cluster. If you encounter issues related to disk quota, please contact the system administrator.

    All scripts are written to be compatible with Python 3.9.12 or higher.
