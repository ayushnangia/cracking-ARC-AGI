# This code is for GPU parallelisation. It works on apple silicon. Minor modifications are needed for other platforms.
# For kaggle or google, you can use the g-prl-nca.py file.

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import datetime
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # Format: YYMMDD_HHMMSS

# --- 1. SETTINGS & PATHS ---
# Uncomment for local mac silicon run
ARC_DATA_DIR = "../dataset/script-tests/grouped-tasks"
OUTPUT_DIR = os.path.join("./runs", f"test_{timestamp}")

# Uncomment for Kaggle
# ARC_DATA_DIR = "/kaggle/input/arc-prize-2024"
# OUTPUT_DIR = "/kaggle/working"

# Uncomment for Google Colab run
# from google.colab import drive
# drive.mount('/content/drive')
# ARC_DATA_DIR = "/content/drive/MyDrive/Cracking-ARC-AGI/dataset/script-tests/grouped-tasks"
# OUTPUT_DIR = os.path.join("/content/drive/MyDrive/Cracking-ARC-AGI/NCAs", f"test_{timestamp}")

INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, "challenges.json")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
HPARAMS: Dict[str, Any] = {
    "grid_size": 30,
    "n_classes": 11,
    "in_channels": 20, # 11 for color one-hot, 5 for hidden state
    "hidden_channels": 9,
    "nn_hidden_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_iterations": 400,
    "prediction_steps": 30,
    "train_steps_min": 30,
    "train_steps_max": 30
}

N_WORKERS = 5

# --- 2. The CellularNN Model ---

class CellularNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        
        # For a 3x3 neighborhood, there are 8 neighbors.
        perception_channels = self.in_channels + (8 * self.n_classes) 
        
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
        """
        Alternative implementation to get neighbor states using padding and slicing,
        aiming to avoid F.unfold and its potential MPS fallback for 'col2im'.
        """
        # We only care about the one-hot color state of neighbors
        color_state = x[:, :self.n_classes]  # Shape: [B, n_classes, H, W]
        B, C_state, H, W = color_state.shape

        # Pad the color_state to handle boundaries easily.
        # Padding is (pad_left, pad_right, pad_top, pad_bottom) for the last two dims.
        padded_color_state = F.pad(color_state, (1, 1, 1, 1), mode='constant', value=0.0)
        # padded_color_state shape: [B, n_classes, H+2, W+2]

        neighbor_tensors = []

        # Iterate through the 3x3 neighborhood offsets, skipping the center (0,0)
        for r_offset in range(-1, 2):  # -1, 0, 1
            for c_offset in range(-1, 2):  # -1, 0, 1
                if r_offset == 0 and c_offset == 0:
                    continue  # Skip the center cell itself

                start_row = r_offset + 1
                end_row = start_row + H
                start_col = c_offset + 1
                end_col = start_col + W

                neighbor_slice = padded_color_state[:, :, start_row:end_row, start_col:end_col]
                neighbor_tensors.append(neighbor_slice)

        all_neighbors = torch.cat(neighbor_tensors, dim=1)
        return all_neighbors


    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        neighbor_channels = self.get_neighbor_states(x)
        return torch.cat([x, neighbor_channels], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        perception = self.perceive(x)
        h = F.relu(self.fc1(perception))
        dx = self.fc2(h)
        new_state = x + dx
        
        # Apply LayerNorm over the C dimension at each spatial location
        x_perm = new_state.permute(0, 2, 3, 1)
        x_norm = self.layernorm(x_perm)
        new_state = x_norm.permute(0, 3, 1, 2)
        return new_state

# --- 3. Helper Functions ---

def create_array_from_grid(
    small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int
) -> np.ndarray:
    """Creates a (grid_size, grid_size, in_channels) numpy array from a small grid."""
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    # Channel 0 is the "empty" channel
    arr[:, :, 0] = 1.0

    small_grid_np = np.array(small_grid, dtype=np.int32)
    rows, cols = small_grid_np.shape
    max_rows, max_cols = min(rows, grid_size), min(cols, grid_size)

    for i in range(max_rows):
        for j in range(max_cols):
            pixel_val = small_grid_np[i, j]
            # Colors 0-9 map to channels 1-10
            if 0 <= pixel_val <= (n_classes - 2):
                arr[i, j, :n_classes] = 0.0
                arr[i, j, pixel_val + 1] = 1.0
    return arr

def tensor_to_grid(state_tensor: torch.Tensor, n_classes: int) -> List[List[int]]:
    """Converts a single predicted state tensor [C, H, W] to a List[List[int]] grid."""
    pred_indices = state_tensor.cpu()[:n_classes, :, :].argmax(dim=0).numpy()
    # Convert channel indices back to ARC color values. Channel 0 is empty, Channel 1 is color 0, etc.
    grid = (pred_indices - 1).tolist()
    # Replace -1 (from channel 0) with 0.
    # return [[max(0, cell) for cell in row] for row in grid]
    return grid

def depad_grid(grid: List[List[int]], padding_value: int = -1) -> List[List[int]]:
    """Removes padding from a grid by finding the smallest bounding box containing non-padding values."""
    if not grid or not grid[0]:
        return [[padding_value]]

    rows = len(grid)
    cols = len(grid[0])
    min_r, max_r, min_c, max_c = -1, -1, cols, -1
    found_non_padding = False

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx][c_idx] != padding_value:
                if not found_non_padding:
                    min_r = r_idx
                max_r = r_idx
                min_c = min(min_c, c_idx)
                max_c = max(max_c, c_idx)
                found_non_padding = True
    
    if not found_non_padding:
        return [[padding_value]]

    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]

# --- 4. The "Workhorse" Function ---

def train_and_predict_for_task(
    task_id: str, 
    train_pairs: List[Dict], 
    test_inputs: List[Dict], 
    device: torch.device, 
    hparams: Dict[str, Any],
    process_id: int = 0  # To distinguish logs if needed
) -> Tuple[str, List[List[int]]]: # Return a tuple
    """
    Initializes, trains, and uses an NCA model for a single task.
    Returns a tuple of (task_id, list_of_predicted_grids).
    """
    # 1. Model & Optimizer Initialization
    model = CellularNN(
        hparams['in_channels'], hparams['n_classes'], hparams['nn_hidden_dim']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay']
    )

    # 2. Data Preparation
    grid_args = (hparams['grid_size'], hparams['in_channels'], hparams['n_classes'])
    train_input_tensors = [
        torch.tensor(create_array_from_grid(p['input'], *grid_args)).permute(2, 0, 1) 
        for p in train_pairs
    ]
    train_target_tensors = [
        torch.tensor(create_array_from_grid(p['output'], *grid_args)).permute(2, 0, 1)
        for p in train_pairs
    ]

    # 3. Training Loop
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
        loss = F.cross_entropy(state[:, :hparams['n_classes']], target_labels)
        loss.backward()
        optimizer.step()

        # Logging is less clean in parallel, but can be useful for debugging
        # if i % 200 == 0:
        #     print(f"  [Worker {process_id}] Task {task_id}, Iter {i:04d}: Loss={loss.item():.4f}")
            
    # 4. Save Final Checkpoint
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{task_id}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)

    # 5. Prediction Phase
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

def worker_process(args):
    """
    A simple wrapper to unpack arguments for pool.starmap and call the main function.
    """
    task_id, train_pairs, test_inputs, device, hparams, process_id = args
    # print(f"Worker {process_id} starting task {task_id}") # Uncomment for debug
    
    if not train_pairs:
        # Handle tasks with no training pairs
        predicted_grids = [[[0]] for _ in test_inputs]
        return (task_id, predicted_grids)
    else:
        try:
            return train_and_predict_for_task(
                task_id, train_pairs, test_inputs, device, hparams, process_id
            )
        except Exception as e:
            print(f"!!! ERROR processing task {task_id}: {e}")
            # Return a default failure prediction
            predicted_grids = [[[0]] for _ in test_inputs]
            return (task_id, predicted_grids)

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    script_start_time = time.time()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {N_WORKERS} parallel workers.")

    # Load data
    try:
        with open(INPUT_JSON_FILE, 'r') as f:
            challenges = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Input JSON not found at {INPUT_JSON_FILE}")
        exit()
    
    # Prepare arguments for all tasks
    task_args = []
    task_ids = list(challenges.keys())
    for i, task_id in enumerate(task_ids):
        task_data = challenges[task_id]
        train_pairs = task_data['train']
        test_inputs = task_data['test']
        # Each task gets its arguments packed into a tuple
        task_args.append((task_id, train_pairs, test_inputs, device, HPARAMS, i % N_WORKERS))

    # Main parallel processing pool
    submission = {}
    
    # The 'spawn' start method is often more compatible with CUDA
    # You might not need this on macOS/Linux but it's good practice
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)

    with Pool(processes=N_WORKERS) as pool:
        # Use tqdm to show a progress bar
        results = list(tqdm(pool.imap_unordered(worker_process, task_args), total=len(task_args)))

    # Process results and format submission
    for task_id, predicted_grids in results:
        formatted_predictions = []
        for grid in predicted_grids:
            formatted_predictions.append({
                "attempt_1": grid,
                "attempt_2": grid
            })
        submission[task_id] = formatted_predictions
        print(f"  Finished processing and collating results for task {task_id}")


    # Save final submission file
    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f)

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    print("\n-----------------------------------------")
    print(f"Success! Submission file saved to {SUBMISSION_FILE}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("-----------------------------------------")