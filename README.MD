
## Setup and Installation

This project uses `uv` for fast and reliable dependency management.

1.  **Install `uv`**:
    If you don't have `uv` installed, follow the official instructions. For most systems:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Virtual Environment**:
    From the project's root directory, create and activate a virtual environment:
    ```bash
    # Create the virtual environment in a .venv folder
    uv venv --seed

    # Activate it (on Linux/macOS)
    source .venv/bin/activate
    # On Windows (Powershell):
    # .venv\Scripts\Activate.ps1
    ```

3.  **Dependency solve that thang!**:
    `uv sync --extra cuda`

4.  **run it!**:
    ~~`python bench_v_pred.py`~~
    better logging, less interactive repl:
    `python bench_multires_cl.py`

    `python diffusion_lab_bench_v3`
    to concretely measure gradient 

    `python compiler_daemon.py`
    for crimes against torch

    `python bench_backbone_hit_that_griddy.py`
    ???