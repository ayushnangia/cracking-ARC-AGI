# Many plain vanilla NCA models combined into a large Multi-NCA model to get parallelisation.
# No GPU shenanigans.
# This gives a speedup on colab, but not on local mac silicon. Todo: Figure out why. 

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Tuple
import datetime
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # Format: YYMMDD_HHMMSS

# --- 1. SETTINGS & PATHS (EDIT THESE) ---
# For local run
ARC_DATA_DIR = "../../dataset/script-tests/grouped-tasks-0-4x" # Adjusted for typical local setup
OUTPUT_DIR = os.path.join("./runs", f"parallel_nca_{timestamp}")

# Uncomment for Kaggle
# ARC_DATA_DIR = "/kaggle/input/arc-prize-2024"
# OUTPUT_DIR = "/kaggle/working"

# Uncomment for Colab
# from google.colab import drive
# drive.mount('/content/drive')
# ARC_DATA_DIR = "/content/drive/MyDrive/Cracking-ARC-AGI/dataset/script-tests/grouped-tasks-0-4x"
# OUTPUT_DIR = os.path.join("/content/drive/MyDrive/Cracking-ARC-AGI/NCAs", f"multi-test_{timestamp}")
# VISUALISE = False   # Set to True to generate visualization.pdf at the end of execution
# EVALUATE_SCRIPT_PATH = "/content/drive/MyDrive/Cracking-ARC-AGI/evaluate.py"

INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, "challenges.json")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters - Aligned with nca5.py for benchmarking
HPARAMS: Dict[str, Any] = {
    "min_train_pairs": 5, # Only train tasks with at least this many training pairs
    "grid_size": 30,
    "n_classes": 11,
    "in_channels": 20, # 11 for color one-hot, 9 for hidden state
    "hidden_channels": 9,
    "nn_hidden_dim": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_iterations": 400,
    "batch_size_per_task": 5, # BATCH_SIZE for each task when forming a mega-batch
    "prediction_steps": 30,
    "train_steps_min": 30,
    "train_steps_max": 30,
    "max_norm": 1.0
}

# --- 2. The Fully ParallelNCA Model ---

class ParallelNCA(nn.Module):
    def __init__(self, num_models: int, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.num_models = num_models
        self.in_channels = in_channels
        self.n_classes = n_classes

        # In nca5, perception uses all channels from neighbors
        perception_channels_per_model = self.in_channels * 9

        # Grouped convolutions for parallel processing of all models
        self.fc1 = nn.Conv2d(
            self.num_models * perception_channels_per_model,
            self.num_models * nn_hidden_dim,
            1,
            groups=self.num_models
        )
        self.fc2 = nn.Conv2d(
            self.num_models * nn_hidden_dim,
            self.num_models * self.in_channels,
            1,
            groups=self.num_models
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for the grouped convolutions
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight) # Critical: init output layer weights to zero for stable start
        nn.init.zeros_(self.fc2.bias)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, M * C_in, H, W]
        # We use all channels of neighbors, same as nca5.py
        padded_state = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.0)
        
        neighbor_tensors = []
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                if r_offset == 0 and c_offset == 0:
                    continue
                start_row, end_row = r_offset + 1, r_offset + 1 + x.size(2)
                start_col, end_col = c_offset + 1, c_offset + 1 + x.size(3)
                neighbor_tensors.append(padded_state[:, :, start_row:end_row, start_col:end_col])
        
        neighbor_channels = torch.cat(neighbor_tensors, dim=1) # Shape: [B, M * 8 * C_in, H, W]
        return torch.cat([x, neighbor_channels], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, M * C_in, H, W]
        perception = self.perceive(x)
        h = F.relu(self.fc1(perception))
        dx = self.fc2(h)
        new_state = x + dx
        # No normalization, to match nca5.py
        return new_state

    def get_specific_model_output(self, x_single_batch: torch.Tensor, model_idx: int) -> torch.Tensor:
        # For prediction, we run a single model on a single input batch
        # x_single_batch shape: [B, C_in, H, W]
        
        # Extract the weights and biases for the specific model
        def get_grouped_params(layer, model_idx):
            out_ch_total, in_ch_per_group, k, _ = layer.weight.shape
            out_ch_per_model = out_ch_total // self.num_models

            w_start_out = model_idx * out_ch_per_model
            w_end_out = (model_idx + 1) * out_ch_per_model
            
            # Correctly slice the weights for the model. 
            # The input channels (dim=1) are already per-group for a grouped convolution.
            weight = layer.weight[w_start_out:w_end_out, :, :, :].clone()
            
            b_start = model_idx * out_ch_per_model
            b_end = (model_idx + 1) * out_ch_per_model
            bias = layer.bias[b_start:b_end].clone()
            
            return weight, bias

        w1, b1 = get_grouped_params(self.fc1, model_idx)
        w2, b2 = get_grouped_params(self.fc2, model_idx)

        # --- Recreate the forward pass for a single model ---
        # 1. Perception
        padded_state = F.pad(x_single_batch, (1, 1, 1, 1), mode='constant', value=0.0)
        neighbor_tensors = []
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                if r_offset == 0 and c_offset == 0: continue
                
                start_row, end_row = r_offset + 1, r_offset + 1 + x_single_batch.size(2)
                start_col, end_col = c_offset + 1, c_offset + 1 + x_single_batch.size(3)
                neighbor_tensors.append(padded_state[:, :, start_row:end_row, start_col:end_col])

        neighbor_channels = torch.cat(neighbor_tensors, dim=1)
        perception_single = torch.cat([x_single_batch, neighbor_channels], dim=1)
        
        # 2. Convolutions
        h = F.relu(F.conv2d(perception_single, w1, b1))
        dx = F.conv2d(h, w2, b2)

        # 3. Update (no normalization)
        new_state = x_single_batch + dx
        return new_state


# --- 3. Helper Functions ---

def create_array_from_grid(
    small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int
) -> np.ndarray:
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    # Initialize color channel 0 (empty) to 1.0 for all cells
    arr[:, :, 0] = 1.0
    # Hidden channels are already 0.0 by np.zeros

    small_grid_np = np.array(small_grid, dtype=np.int32)
    rows, cols = small_grid_np.shape
    max_rows, max_cols = min(rows, grid_size), min(cols, grid_size)

    for i in range(max_rows):
        for j in range(max_cols):
            pixel_val = small_grid_np[i, j]
            # ARC colors 0-9 map to one-hot channels 1-10
            if 0 <= pixel_val <= (n_classes - 2): # Max valid color value
                arr[i, j, :n_classes] = 0.0       # Reset all color channels for this pixel
                arr[i, j, pixel_val + 1] = 1.0    # Set the specific color channel
            # Else, it remains "empty" (channel 0 is 1.0)
    return arr

def tensor_to_grid(state_tensor: torch.Tensor, n_classes: int) -> List[List[int]]:
    # state_tensor: [C, H, W]
    # Take argmax over the n_classes color channels
    pred_indices = state_tensor.cpu()[:n_classes, :, :].argmax(dim=0).numpy()
    # Map one-hot channel index back to ARC color value:
    # Channel 0 -> color -1 
    # Channel 1 -> color 0
    grid = (pred_indices - 1).tolist() 
    return grid

def depad_grid(grid: List[List[int]], padding_value: int = -1) -> List[List[int]]:
    if not grid or not grid[0]: return [[padding_value]]

    rows = len(grid)
    cols = len(grid[0])
    min_r, max_r, min_c, max_c = -1, -1, cols, -1
    found_non_padding = False

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx][c_idx] != padding_value:
                if not found_non_padding: min_r = r_idx
                max_r = max(max_r, r_idx)
                min_c = min(min_c, c_idx)
                max_c = max(max_c, c_idx)
                found_non_padding = True
    
    if not found_non_padding: return [[padding_value]]

    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]

# --- 4. The "Workhorse" Function ---

def train_and_predict_parallel(
    all_tasks_data: Dict[str, Dict], 
    active_task_ids: List[str],     
    task_id_to_model_idx: Dict[str, int], # Important for mapping
    device: torch.device,
    hparams: Dict[str, Any]
) -> Dict[str, List[List[List[int]]]]:

    grid_init_args = (
        hparams['grid_size'], 
        hparams['in_channels'], 
        hparams['n_classes']
    )
    prepared_train_data = {} 

    for task_id in active_task_ids:
        train_pairs = all_tasks_data[task_id]['train']
        prepared_train_data[task_id] = {
            'inputs': [
                torch.tensor(create_array_from_grid(p['input'], *grid_init_args)).permute(2, 0, 1)
                for p in train_pairs
            ],
            'targets': [ 
                torch.tensor(create_array_from_grid(p['output'], *grid_init_args)).permute(2, 0, 1)
                for p in train_pairs
            ]
        }

    num_active_models = len(active_task_ids)
    if num_active_models == 0:
        print("No active tasks with training data. Returning default predictions.")
        all_predictions = {}
        for task_id, task_content in all_tasks_data.items():
            all_predictions[task_id] = [[[0]] for _ in task_content['test']]
        return all_predictions

    # The single, parallel model
    parallel_nca_model = ParallelNCA(
        num_models=num_active_models,
        in_channels=hparams['in_channels'],
        n_classes=hparams['n_classes'],
        nn_hidden_dim=hparams['nn_hidden_dim']
    ).to(device)

    optimizer = torch.optim.Adam(
        parallel_nca_model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay']
    )

    batch_size_per_task = hparams['batch_size_per_task']

    print(f"Starting training for {num_active_models} tasks in parallel with {hparams['num_iterations']} iterations.")
    parallel_nca_model.train()
    for i in range(hparams['num_iterations']):
        # Batch sampling and reshaping for parallel model
        mega_batch_inputs_list = [[] for _ in range(batch_size_per_task)]
        mega_batch_targets_list = [[] for _ in range(batch_size_per_task)]

        for model_idx, task_id in enumerate(active_task_ids): 
            task_train_inputs = prepared_train_data[task_id]['inputs']
            task_train_targets = prepared_train_data[task_id]['targets']
            
            batch_indices = np.random.choice(
                len(task_train_inputs), batch_size_per_task, replace=True
            )

            for batch_slot, sample_idx in enumerate(batch_indices):
                mega_batch_inputs_list[batch_slot].append(task_train_inputs[sample_idx])
                mega_batch_targets_list[batch_slot].append(task_train_targets[sample_idx])
        
        # Concatenate along channel dimension for the parallel model
        inp_megabatch = torch.stack([torch.cat(batch, dim=0) for batch in mega_batch_inputs_list]).to(device)
        target_megabatch = torch.stack([torch.cat(batch, dim=0) for batch in mega_batch_targets_list]).to(device)
        
        optimizer.zero_grad()

        state_megabatch = inp_megabatch
        num_nca_steps = np.random.randint(hparams['train_steps_min'], hparams['train_steps_max'] + 1)
        for _ in range(num_nca_steps):
            state_megabatch = parallel_nca_model(state_megabatch) 
        
        # Loss calculation: MSE on all channels, to match nca5.py
        loss = F.mse_loss(state_megabatch, target_megabatch)
        
        loss.backward()
        # Gradient clipping, to match nca5.py
        torch.nn.utils.clip_grad_norm_(parallel_nca_model.parameters(), hparams['max_norm'])
        optimizer.step()

        if i % 200 == 0:
            print(f"  Iter {i:04d}/{hparams['num_iterations']}: Mega-Loss={loss.item():.4f}")
            
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"parallel_nca_model_{timestamp}.pth")
    torch.save(parallel_nca_model.state_dict(), final_checkpoint_path)
    print(f"  Saved final parallel NCA model to {final_checkpoint_path}")

    print("\nStarting prediction phase...")
    parallel_nca_model.eval()
    all_task_predictions = {} 

    with torch.no_grad():
        # Iterate over all tasks from original file to produce submission
        for original_task_id in all_tasks_data.keys():
            task_content = all_tasks_data[original_task_id]
            test_inputs_for_task = task_content['test']
            task_predicted_grids = []
            
            if original_task_id not in task_id_to_model_idx:
                task_predicted_grids = [[[0]] for _ in test_inputs_for_task]
            else:
                model_idx = task_id_to_model_idx[original_task_id]
                
                # print(f"  Predicting for task {original_task_id} (model {model_idx+1}/{num_active_models})")
                
                for test_case in test_inputs_for_task:
                    test_input_grid = test_case['input']
                    inp_array = create_array_from_grid(test_input_grid, *grid_init_args)
                    inp_tensor = torch.tensor(inp_array).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    current_state = inp_tensor
                    for _ in range(hparams['prediction_steps']):
                        current_state = parallel_nca_model.get_specific_model_output(current_state, model_idx)

                    grid_from_tensor = tensor_to_grid(current_state.squeeze(0), hparams['n_classes'])
                    depadded_grid = depad_grid(grid_from_tensor, padding_value=-1) 
                    # Convert any remaining internal -1s (empty) to 0s (black), same as nca5
                    final_output_grid = [[max(0, cell) for cell in row] for row in depadded_grid]
                    task_predicted_grids.append(final_output_grid)
            
            all_task_predictions[original_task_id] = task_predicted_grids
            
    return all_task_predictions

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    script_start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Hyperparameters: {json.dumps(HPARAMS, indent=2)}")

    try:
        with open(INPUT_JSON_FILE, 'r') as f:
            all_challenges_data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Input JSON not found at {INPUT_JSON_FILE}")
        exit()
    
    submission = {}
    all_task_ids_from_file = list(all_challenges_data.keys()) # Keep original order for final submission
    print(f"Found {len(all_task_ids_from_file)} tasks in {INPUT_JSON_FILE}")

    # active_task_ids determines which models are created and their order in the ParallelNCA
    # Sorting ensures consistent model order if tasks are re-read
    active_task_ids = sorted([
        task_id for task_id, data in all_challenges_data.items()
        if data.get('train') and len(data['train']) >= HPARAMS['min_train_pairs']
    ])
    
    if not active_task_ids:
        print("No tasks with training data found. Generating default submission for all tasks.")
        for task_id_key in all_task_ids_from_file:
            num_test_cases = len(all_challenges_data[task_id_key]['test'])
            submission[task_id_key] = [
                {"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(num_test_cases)
            ]
    else:
        print(f"Identified {len(active_task_ids)} active tasks with >= {HPARAMS['min_train_pairs']} pairs for parallel training.")
        # Create a mapping from task ID to its index (model_idx) in the parallel model
        task_id_to_model_idx = {task_id: i for i, task_id in enumerate(active_task_ids)}
        
        task_predictions_map = train_and_predict_parallel(
            all_tasks_data=all_challenges_data,
            active_task_ids=active_task_ids,
            task_id_to_model_idx=task_id_to_model_idx,
            device=device,
            hparams=HPARAMS
        )

        # Ensure submission is in the same order as input file tasks
        for task_id_key in all_task_ids_from_file:
            predicted_grids_for_task = task_predictions_map.get(task_id_key, [])
            if not predicted_grids_for_task: # Should only happen if task_id_key wasn't in map (e.g. error)
                 num_test_cases = len(all_challenges_data[task_id_key]['test'])
                 predicted_grids_for_task = [[[0]] for _ in range(num_test_cases)]


            formatted_predictions = []
            for grid in predicted_grids_for_task:
                formatted_predictions.append({"attempt_1": grid, "attempt_2": grid})
            submission[task_id_key] = formatted_predictions

    with open(SUBMISSION_FILE, 'w') as f:
        json.dump(submission, f, indent=2)

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    print("\n-----------------------------------------")
    print(f"Success! Submission file saved to {SUBMISSION_FILE}")
    print(f"Total execution time: {total_time:.2f} seconds")
    if active_task_ids:
        print(f"Trained {len(active_task_ids)} tasks in parallel.")
    else:
        print("No tasks were trained.")
    print("-----------------------------------------")