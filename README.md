# Text -> Handwriting
1. Train a general diffusion model to generate handwriting 
2. LoRA the model to make it generate handwriting in specific styles.

## Implementation
* Architecture: https://arxiv.org/pdf/2011.06704 [repo](https://github.com/tcl9876/Diffusion-Handwriting-Generation/tree/master)
* Dataset: [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database)

LoRA code and dataset coming later, let me cook.

## Usage
1. Download the dataset from IAM,  put it into the ``data`` directory
2. Make sure the python version is 3.12.0, use whatever environment manager (I prefer python -m venv env) and download requirements.txt
3. Run preprocessing.py, then run train.py
