## NOTETIFY
Use virtual env for packages

# Crate venv
### NOTE: Python version <= 3.12 for this project to work
python -m venv env

# Run venv
source env/Scripts/activate
deactivate

# Training
To run training run python3 cnn_training.py


# BEFORE you run this make sure that you are in a venv for this project
# Run requirments.txt for packages
pip install -r requirements.txt

## DEBUG_MODE
export DEBUG_MODE = True (Default False)
$DEBUG_MODE (Check current value)