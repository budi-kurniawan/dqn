# Creating and Activating a venv
    python3 -m venv venv

# Activate it
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate

    # 3. Now install Gymnasium



Installing All Dependencies 
    pip install -r requirements.txt


Installing dependencies for the user (to avoid using venv):
    pip install --user --break-system-packages gymnasium matplotlib torch

# Running the App
Make sure you have a pyproject.toml
From the project root, run
    pip install -e .
    python -m agent.main

# GPU Profiling
    nvidia-smi dmon -s uc

# Variants
- dqn: adaptation from PyTorch website where Transition contains tensors
- dqn2: adaptation from dqn where Transition contains Numpy's ndarrays and DQNAgent's optimize_model() avoids tensor operations
- torchdqn: uses TorchCartpole and Torch tensors throughout, run using torchmain.py
- mtorchdqn.py, A variant of torchdqn that allows multiple environments