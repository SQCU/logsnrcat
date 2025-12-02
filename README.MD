
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
    test marginal benefit of different flops assignments

5. **why low resolution checkerboards?**
    technically speaking this project is testing 2 hard things:

    1: generating data without supervisory signals like a perceptual or GAN loss or conditioning models

    2: at multiple different image scales *concurrently* across pretraining, which breaks many feature extraction and visual learning backbones

    the checkerboard patterns are the simplest synthetic data i could imagine in 1 minute which provides examples of most of the values the R,G,B colorspace can take, where there are both local features (edges, contiguous shapes) and global features (color pairs, parallel lines??) which don't need a supervisory model to measure and interpret.

    you can, and should, calculate the frechet distance of the color histograms found in your output samples vs the color histograms in your generated training samples!
    generalization of the frechet distance over parameter-free metrics to spatial correlation and parallelism/contiguity of color regions is left as an exercise to whoever forks the project next.

6. **wait so does all of the stuff in this repo work?**
    hell yeah it does!

    even the weird torch kernel compilation interceptor server-client pair. i dont recommend using it (easy 2 comment out) but i don't recommend *not* using it either.