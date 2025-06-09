# =================================================================================================================
#  SINGLE-FILE UNIFIED PARALLEL NCA RUNNER (KAGGLE / COLAB / LOCAL MAC)
# =================================================================================================================
# This script is a self-contained, portable solution for running parallelized NCA tasks.
#
# HOW TO USE:
# 1. Scroll down to the `if __name__ == "__main__":` block.
# 2. Find the "--- PLATFORM CONFIGURATION ---" section.
# 3. Uncomment the block for the environment you are using (KAGGLE, GOOGLE COLAB, or LOCAL_MAC).
# 4. Ensure all other platform blocks are commented out.
# 5. Run the script. It will automatically detect the best available hardware (CUDA, MPS, CPU).
# =================================================================================================================

import os
import json
import time
import torch
import sys
from typing import List, Dict, Any, Tuple
import datetime
import multiprocessing as mp
from tqdm import tqdm

# =================================================================================================================
# | SECTION 1: WORKER CODE                                                                                        |
# | This entire block of code will be written to a file at runtime. It is fully self-contained.                   |
# =================================================================================================================

WORKER_CODE = """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

# 1. The CellularNN Model
class CellularNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        perception_channels = self.in_channels + (8 * self.in_channels)
        self.fc1 = nn.Conv2d(perception_channels, nn_hidden_dim, 1)
        self.fc2 = nn.Conv2d(nn_hidden_dim, self.in_channels, 1)
        self.layernorm = nn.LayerNorm(self.in_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def get_neighbor_states(self, x: torch.Tensor) -> torch.Tensor:
        padded_state = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
        B, C_state, H, W = x.shape
        neighbor_tensors = []
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                if r_offset == 0 and c_offset == 0: continue
                neighbor_slice = padded_state[:, :, 1+r_offset:H+1+r_offset, 1+c_offset:W+1+c_offset]
                neighbor_tensors.append(neighbor_slice)
        return torch.cat(neighbor_tensors, dim=1)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        neighbor_channels = self.get_neighbor_states(x)
        return torch.cat([x, neighbor_channels], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        perception = self.perceive(x)
        h = F.relu(self.fc1(perception))
        dx = self.fc2(h)
        new_state = x + dx
        x_perm = new_state.permute(0, 2, 3, 1)
        x_norm = self.layernorm(x_perm)
        new_state = x_norm.permute(0, 3, 1, 2)
        return new_state

# 2. Helper Functions
def create_array_from_grid(small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int) -> np.ndarray:
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    arr[:, :, 0] = 1.0
    if not small_grid or not small_grid[0]: return arr
    small_grid_np = np.array(small_grid, dtype=np.int32)
    rows, cols = small_grid_np.shape
    for i in range(min(rows, grid_size)):
        for j in range(min(cols, grid_size)):
            pixel_val = small_grid_np[i, j]
            if 0 <= pixel_val <= (n_classes - 2):
                arr[i, j, :n_classes] = 0.0
                arr[i, j, pixel_val + 1] = 1.0
    return arr

def tensor_to_grid(state_tensor: torch.Tensor, n_classes: int) -> List[List[int]]:
    pred_indices = state_tensor.cpu()[:n_classes, :, :].argmax(dim=0).numpy()
    grid = (pred_indices - 1).tolist()
    return grid

def depad_grid(grid: List[List[int]], padding_value: int = -1) -> List[List[int]]:
    if not grid or not grid[0]: return [[padding_value]]
    rows, cols = len(grid), len(grid[0])
    min_r, max_r, min_c, max_c = -1, -1, cols, -1
    found = False
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != padding_value:
                if not found: min_r = r
                max_r, min_c, max_c = r, min(min_c, c), max(max_c, c)
                found = True
    if not found: return [[padding_value]]
    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]

# 3. The Workhorse Function
def train_and_predict_for_task(
    task_id: str,
    train_pairs: List[Dict],
    test_inputs: List[Dict],
    device_str: str,
    hparams: Dict[str, Any]
) -> Tuple[str, List[List[int]]]:

    device = torch.device(device_str)
    model = CellularNN(hparams['in_channels'], hparams['n_classes'], hparams['nn_hidden_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    grid_args = (hparams['grid_size'], hparams['in_channels'], hparams['n_classes'])
    train_input_tensors = [torch.tensor(create_array_from_grid(p['input'], *grid_args)).permute(2, 0, 1) for p in train_pairs]
    train_target_tensors = [torch.tensor(create_array_from_grid(p['output'], *grid_args)).permute(2, 0, 1) for p in train_pairs]

    model.train()
    for i in range(hparams['num_iterations']):
        inp_batch = torch.stack(train_input_tensors).to(device)
        target_batch = torch.stack(train_target_tensors).to(device)
        optimizer.zero_grad()
        state = inp_batch
        num_steps = np.random.randint(hparams['train_steps_min'], hparams['train_steps_max'] + 1)
        for _ in range(num_steps):
            state = model(state)
        target_labels = target_batch.argmax(dim=1)
        loss_ce = F.cross_entropy(state[:, :hparams['n_classes']], target_labels)
        loss_mse = F.mse_loss(state[:, hparams['n_classes']:], target_batch[:, hparams['n_classes']:])
        loss = loss_ce + loss_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['max_norm'])
        optimizer.step()

    model.eval()
    predicted_grids = []
    with torch.no_grad():
        for test_case in test_inputs:
            inp_array = create_array_from_grid(test_case['input'], *grid_args)
            inp_tensor = torch.tensor(inp_array).permute(2, 0, 1).unsqueeze(0).to(device)
            state = inp_tensor
            for _ in range(hparams['prediction_steps']):
                state = model(state)
            grid_from_tensor = tensor_to_grid(state.squeeze(0), hparams['n_classes'])
            depadded_grid = depad_grid(grid_from_tensor)
            final_output_grid = [[max(0, cell) for cell in row] for row in depadded_grid]
            predicted_grids.append(final_output_grid)

    return (task_id, predicted_grids)

# 4. The worker wrapper function for the pool
def worker_process(args):
    task_id, train_pairs, test_inputs, device_str, hparams = args
    if not train_pairs:
        predicted_grids = [[[0]] for _ in test_inputs]
        return (task_id, predicted_grids)
    else:
        try:
            return train_and_predict_for_task(
                task_id, train_pairs, test_inputs, device_str, hparams
            )
        except Exception as e:
            # It's good practice to print the specific task and device that failed.
            print(f"!!! ERROR processing task {task_id} on device {device_str}: {e}", flush=True)
            predicted_grids = [[[0]] for _ in test_inputs]
            return (task_id, predicted_grids)
"""

# =================================================================================================================
# | SECTION 2: MAIN EXECUTION SCRIPT                                                                              |
# =================================================================================================================

if __name__ == "__main__":
    script_start_time = time.time()

    # --- PLATFORM CONFIGURATION: UNCOMMENT THE BLOCK FOR YOUR CURRENT ENVIRONMENT ---

    # # --- 1. KAGGLE CONFIG ---
    # PLATFORM_NAME = "Kaggle"
    # ARC_DATA_DIR = "/kaggle/input/arc1-train-grouped"
    # INPUT_JSON_FILENAME = "challenges.json"
    # OUTPUT_DIR = "/kaggle/working/"
    # WORKERS_PER_GPU = 5 # Rule: 5 workers per L4 GPU.
    # LOCAL_WORKERS = 20 # Fallback for CPU/MPS, Kaggle typically has many cores
    # VISUALISE = True   # Set to True to generate visualization.pdf at the end of execution
    # EVALUATE_SCRIPT_PATH = "/kaggle/working/evaluate.py" # Adjust if evaluate.py is in a dataset

    # # --- 2. GOOGLE COLAB CONFIG ---
    # PLATFORM_NAME = "Google Colab"
    # # NOTE: Make sure to mount your Google Drive before running!
    # from google.colab import drive
    # drive.mount('/content/drive')
    # ARC_DATA_DIR = "/content/drive/MyDrive/Cracking-ARC-AGI/dataset/script-tests/grouped-tasks-4"
    # INPUT_JSON_FILENAME = "challenges.json"
    # OUTPUT_DIR = os.path.join("/content/drive/MyDrive/Cracking-ARC-AGI/NCAs/runs", datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
    # WORKERS_PER_GPU = 5 # Same rule, but will likely only find 1 GPU.
    # LOCAL_WORKERS = 5   # Fallback for CPU
    # VISUALISE = True   # Set to True to generate visualization.pdf at the end of execution
    # EVALUATE_SCRIPT_PATH = "/content/drive/MyDrive/Cracking-ARC-AGI/evaluate.py"

    # --- 3. LOCAL MAC/PC CONFIG ---
    PLATFORM_NAME = "Local Mac/PC"
    ARC_DATA_DIR = "../../dataset/script-tests/grouped-tasks"
    INPUT_JSON_FILENAME = "challenges.json"
    OUTPUT_DIR = os.path.join("../runs", f"grun_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}")
    WORKERS_PER_GPU = 2 # Used for MPS or if a single CUDA GPU is found locally
    LOCAL_WORKERS = 2   # Used for CPU or as the primary setting for MPS
    VISUALISE = True   # Set to True to generate visualization.pdf at the end of execution
    EVALUATE_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluate.py"))

    # ------------------------------------------------------------------------------------

    print(f"Running on platform: {PLATFORM_NAME}")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    WORKER_FILENAME = os.path.join(OUTPUT_DIR, "temp_nca_worker.py")

    # Hyperparameters
    HPARAMS: Dict[str, Any] = {
        "grid_size": 30, 
        "n_classes": 11, 
        "in_channels": 20,
        "nn_hidden_dim": 128, 
        "lr": 1e-3, 
        "weight_decay": 1e-4,
        "num_iterations": 400, 
        "prediction_steps": 30,
        "train_steps_min": 30, 
        "train_steps_max": 30,
        "max_norm": 1.0
    }

    # --- DYNAMIC WORKER FILE CREATION ---
    try:
        with open(WORKER_FILENAME, "w") as f:
            f.write(WORKER_CODE)
        sys.path.insert(0, OUTPUT_DIR)
        from temp_nca_worker import worker_process
        print(f"Successfully wrote and imported worker from {WORKER_FILENAME}")
    except Exception as e:
        print(f"FATAL: Could not write or import the worker file: {e}")
        exit()

    # Set 'spawn' start method for CUDA and general safety
    try:
        if mp.get_start_method() != 'spawn':
             mp.set_start_method('spawn', force=True)
             print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass # Already set

    # --- DYNAMIC DEVICE AND WORKER SCALING ---
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        N_WORKERS = num_gpus * WORKERS_PER_GPU
        print(f"Found {num_gpus} CUDA GPU(s). Using {N_WORKERS} workers ({WORKERS_PER_GPU} per GPU).")
    elif torch.backends.mps.is_available():
        devices = ["mps"]
        N_WORKERS = LOCAL_WORKERS
        print(f"Found Apple MPS. Using 'mps' device with {N_WORKERS} workers.")
    else:
        devices = ["cpu"]
        N_WORKERS = LOCAL_WORKERS
        print(f"No GPU found. Using 'cpu' device with {N_WORKERS} workers.")
    print(f"Tasks will be distributed across: {devices}")

    # --- DATA LOADING AND TASK PREPARATION ---
    INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, INPUT_JSON_FILENAME)
    try:
        with open(INPUT_JSON_FILE, 'r') as f:
            challenges = json.load(f)
        print(f"Loaded {len(challenges)} tasks from {INPUT_JSON_FILE}")
    except FileNotFoundError:
        print(f"FATAL: Input file not found at {INPUT_JSON_FILE}")
        exit()

    task_args = []
    for i, (task_id, task_data) in enumerate(challenges.items()):
        device_str = devices[i % len(devices)]
        task_args.append((
            task_id, task_data.get('train', []), task_data['test'],
            device_str, HPARAMS
        ))

    # --- PARALLEL PROCESSING ---
    submission = {}
    print(f"\nStarting processing for {len(task_args)} tasks with {N_WORKERS} workers...")
    with mp.Pool(processes=N_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(worker_process, task_args), total=len(task_args)))

    # --- PROCESS AND SAVE RESULTS ---
    for task_id, predicted_grids in results:
        # Adopting the submission format from nca.py.
        # This format uses a list of dictionaries, one for each test case,
        # with keys "attempt_1" and "attempt_2".
        formatted_predictions = []
        for grid in predicted_grids:
            formatted_predictions.append({
                "attempt_1": grid,
                "attempt_2": grid
            })
        submission[task_id] = formatted_predictions

    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)

    if VISUALISE:
        print("\nStarting visualization...")
        try:
            dataset_path = os.path.abspath(ARC_DATA_DIR)
            submission_path = os.path.abspath(SUBMISSION_FILE)
            
            cmd = [sys.executable, EVALUATE_SCRIPT_PATH, "--submission_file", submission_path, "--dataset", dataset_path, "--visualize"]
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("Visualization script finished.")
        except Exception as e:
            print(f"Error during visualization: {e}")

    total_time = time.time() - script_start_time
    print("\n-----------------------------------------")
    print(f"Success! Submission file saved to {SUBMISSION_FILE}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print("-----------------------------------------")