# Neural Cellular Automata for ARC-AGI

This repository explores the application of Neural Cellular Automata (NCAs) to the Abstract Reasoning Corpus (ARC-AGI) challenge. It includes various vanilla NCA implementations and scripts for training, prediction, and evaluation.

## Conten

## Setup

Follow these steps to set up the project environment:

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd cracking-ARC-AGI
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Experiments

The core NCA models and training scripts are located in the `nca-code/` directory. The primary script for running experiments is `gpu_prl_nca5.py` (for the V5 model), which trains a separate NCA model for each task in parallel using multiprocessing.

1.  **Navigate to the script's directory:**
    ```bash
    cd nca-code/vanilla-v5/
    ```

2.  **Run the script:**
    ```bash
    python3 gpu_prl_nca5.py
    ```

3.  **Inputs:**
    *   By default, the script uses `dataset/script-tests/grouped-tasks/challenges.json` as the input file containing the ARC tasks. You can modify the `ARC_DATA_DIR` and `INPUT_JSON_FILENAME` variables within the script if you wish to use a different dataset.
    *   Set up the number of workers and GPUs to use based on your system
    *   Set `VISUALISE` to `True` to generate a `visualization.pdf` at the end of execution

4.  **Outputs:**
    *   A new run-specific directory is created under `nca-code/runs/`.
    *   `submission.json`: The predictions for all tasks.
    *   If `VISUALISE` is set to `True`
        - `results.md`: Detailed performance metrics.
        - `visualization.pdf`: A PDF showing the input, prediction, and ground truth for each test case.

## Standalone Evaluation

You can also run the evaluation script independently on an existing `submission.json` file.

1.  **Run the `evaluate.py` script:**
    ```bash
    python3 nca-code/evaluate.py \
        --submission_file nca-code/runs/<run_name>/submission.json \
        --dataset dataset/ARC-1/grouped-tasks/training/ \
        --visualize
    ```

2.  **Arguments:**
    *   `--submission_file`: Path to the `submission.json` to evaluate.
    *   `--dataset`: Path to the corresponding dataset directory (must contain `challenges.json` and `solutions.json`).
    *   `--visualize`: Generates the `visualization.pdf`.

## Model Comparison

The NCA implementations have evolved across different versions (v1 to v5). The core `CellularNN` model within each version's script (`gpu_prl_nca*.py`) differs primarily in perception, normalization, and loss function.

| Feature                 | `vanilla-v1`     | `vanilla-v2`                             | `vanilla-v3`                                                                 | `vanilla-v4`                             | `vanilla-v5`     |
| :---------------------- | :----------------------------------- | :--------------------------------------- | :--------------------------------------------------------------------------- | :--------------------------------------- | :----------------------------------- |
| **Neighbor Perception** | Color channels only                  | All channels (color + hidden)            | All channels (color + hidden)                                                | All channels (color + hidden)            | All channels (color + hidden)        |
| **Layer Normalization** | Present                              | Present                                  | Present                                                                      | Present                                  | **Absent**                           |
| **Loss Function**       | Cross-Entropy on color channels      | Cross-Entropy on color channels          | **Composite:**<br/>- Cross-Entropy (color)<br/>- MSE (hidden)                  | MSE on all channels (color + hidden)     | MSE on all channels (color + hidden) |

**Summary of Evolution:**
*   **v1:** Basic NCA where neighbors only see color state.
*   **v2:** Perception is enhanced to include all channels (hidden states).
*   **v3:** A composite loss was introduced to guide both color and hidden state learning.
*   **v4:** Simplified the loss to a single MSE term across all channels.
*   **v5:** Built on v4 by removing Layer Normalization.

The code for each version can be found in `nca-code/vanilla-v*/`.