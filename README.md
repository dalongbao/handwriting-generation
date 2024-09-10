# Text -> Handwriting
## Implementation
* Architecture: https://arxiv.org/pdf/2011.06704 [repo](https://github.com/tcl9876/Diffusion-Handwriting-Generation/tree/master)
* Dataset: [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database)

## Usage
1. Download the dataset from IAM,  put it into the ``data`` directory
2. Make sure the python version is 3.12.0, use whatever environment manager (I prefer python -m venv env) and download requirements.txt
3. Run preprocessing.py, then run train.py
