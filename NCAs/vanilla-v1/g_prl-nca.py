# This code is for running the NCA on Google Colab in a parallelized manner.
# It works on L4 and A100 GPUs
# Ensure you upload g_nca_worker.py to the correct directory.

'''
Here is how it ran on A100. It seems that 5 workers is the sweet spot. This might differ for L4s
Since on Kaggle, you have 4 L4s, you would have to modify the code. Maybe you can get away with 20 workers
5 tasks on A100
seq: 131s
prl-5w: 35s

10 tasks on A100
seq: 252s
prl-5w: 65s
prl-10w: 63s

20 tasks on A100
seq: 501s
prl-4w: 144s
prl-5w: 125s
prl-6w: 133s
prl-10w: 121s   
prl-20w: 126s
'''

import os
import json
import time
import torch
from typing import List, Dict, Any
import datetime
import multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm

# Cell 1: Setup and Mounting
# from google.colab import drive
# drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/Cracking-ARC-AGI/NCAs/') # Use your actual path

# Cell 2: Now import your code
from g_nca_worker import worker_process

# --- 1. SETTINGS & PATHS (These stay in the notebook) ---
ARC_DATA_DIR = "/content/drive/MyDrive/Cracking-ARC-AGI/dataset/script-tests/grouped-tasks-4" # Adjust to your path
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("/content/drive/MyDrive/Cracking-ARC-AGI/NCAs/runs", f"cpr_test_{timestamp}")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
HPARAMS: Dict[str, Any] = {
    "grid_size": 30,
    "n_classes": 11,
    "in_channels": 20,
    "hidden_channels": 9,
    "nn_hidden_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_iterations": 400, 
    "prediction_steps": 30,
    "train_steps_min": 30,
    "train_steps_max": 30
}

# PARALLELISM SETTINGS
N_WORKERS = 4

# --- Main Execution Block ---

if __name__ == "__main__":
    script_start_time = time.time()

    # Set 'spawn' method for CUDA
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {N_WORKERS} parallel workers.")

    # Load data
    INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, "challenges.json")
    with open(INPUT_JSON_FILE, 'r') as f:
        challenges = json.load(f)

    # Prepare arguments for all tasks
    task_args = []
    for task_id, task_data in challenges.items():
        # NOTE: We only pass data that can be pickled easily.
        # The device object is created in the worker. We pass the string 'cuda'.
        task_args.append((
            task_id,
            task_data['train'],
            task_data['test'],
            device, # Sending the device object itself can sometimes be tricky. Let's try it.
            HPARAMS
        ))

    # Main parallel processing pool
    submission = {}
    with mp.Pool(processes=N_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(worker_process, task_args), total=len(task_args)))

    # Process results
    for task_id, predicted_grids in results:
        submission[task_id] = [{"attempt_1": grid, "attempt_2": grid} for grid in predicted_grids]

    # Save final submission file
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)

    total_time = time.time() - script_start_time
    print("\n-----------------------------------------")
    print(f"Success! Submission file saved to {SUBMISSION_FILE}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("-----------------------------------------")