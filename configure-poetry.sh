#!/bin/bash

# Activate Poetry environment and allow system packages
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
export PYTHONPATH="$PYTHONPATH:/usr/local/lib/python3.12/site-packages"

# Install dependencies with Poetry
poetry install --no-root

# Explicitly add graph-tool to the Python path
export PYTHONPATH="$PYTHONPATH:/usr/lib/python3.12/site-packages"

# Restart the shell to ensure changes take effect
exec "$SHELL"

# #!/bin/bash

# # Activate Poetry environment and allow system packages
# poetry config virtualenvs.create true
# poetry config virtualenvs.in-project true
# export PYTHONPATH="$PYTHONPATH:/usr/local/lib/python3.12/site-packages"

# poetry install