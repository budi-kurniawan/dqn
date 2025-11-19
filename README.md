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