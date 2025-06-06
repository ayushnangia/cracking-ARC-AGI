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

# --- 1. SETTINGS & PATHS (EDIT THESE) ---
# For local run
ARC_DATA_DIR = "../dataset/script-tests/grouped-tasks" # Adjusted for typical local setup
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # Format: YYMMDD_HHMMSS
OUTPUT_DIR = os.path.join("./runs", f"multinca_test_{timestamp}")

# Uncomment for Kaggle
# ARC_DATA_DIR = "/kaggle/input/arc-prize-2024"
# OUTPUT_DIR = "/kaggle/working"

INPUT_JSON_FILE = os.path.join(ARC_DATA_DIR, "challenges.json")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
HPARAMS: Dict[str, Any] = {
    "grid_size": 30,
    "n_classes": 11,  # 0-9 colors (10) + 1 for "empty" one-hot channel index 0
    "hidden_channels_nca_state": 9, # Number of hidden channels in the NCA state
    "nn_hidden_dim": 128, # Hidden dim for the FC layers within CellularNN
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_iterations": 1000, # Total training iterations for the multi-NCA model
    "batch_size_per_task": 15, # BATCH_SIZE for each task when forming a mega-batch
    "prediction_steps": 30,
    "train_steps_min": 30, # Min NCA update steps per training iteration
    "train_steps_max": 30  # Max NCA update steps per training iteration
}
# Derived HPARAMS
HPARAMS["in_channels"] = HPARAMS["n_classes"] + HPARAMS["hidden_channels_nca_state"]


# --- 2. The CellularNN Model (Sub-model) ---

class CellularNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels # Total channels (color + hidden)
        
        # Perception includes self state (all in_channels) + neighbor color states (n_classes per neighbor)
        perception_channels = self.in_channels + (8 * self.n_classes) 
        
        self.fc1 = nn.Conv2d(perception_channels, nn_hidden_dim, 1)
        self.fc2 = nn.Conv2d(nn_hidden_dim, self.in_channels, 1) # Output delta for all in_channels
        self.layernorm = nn.LayerNorm(self.in_channels) # LayerNorm applies to all in_channels
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight) # Critical: init output layer weights to zero for stable start
        nn.init.zeros_(self.fc2.bias)

    def get_neighbor_states(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, in_channels, H, W]
        # We only care about the one-hot color state of neighbors
        color_state = x[:, :self.n_classes]  # Shape: [B, n_classes, H, W]
        
        padded_color_state = F.pad(color_state, (1, 1, 1, 1), mode='constant', value=0.0)
        
        neighbor_tensors = []
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                if r_offset == 0 and c_offset == 0:
                    continue
                start_row, end_row = r_offset + 1, r_offset + 1 + x.size(2)
                start_col, end_col = c_offset + 1, c_offset + 1 + x.size(3)
                neighbor_tensors.append(padded_color_state[:, :, start_row:end_row, start_col:end_col])
        
        all_neighbors = torch.cat(neighbor_tensors, dim=1) # Shape: [B, 8 * n_classes, H, W]
        return all_neighbors

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
        new_state_norm = x_norm.permute(0, 3, 1, 2) 
        return new_state_norm

# --- 3. The MultiCellularNN Model (Container Model) ---

class MultiCellularNN(nn.Module):
    def __init__(self, num_models: int, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList(
            [CellularNN(in_channels, n_classes, nn_hidden_dim) for _ in range(num_models)]
        )

    def forward(self, x_megabatch: torch.Tensor, batch_size_per_model: int) -> torch.Tensor:
        outputs = []
        if self.num_models == 0 or batch_size_per_model == 0: # Handle edge cases
             return torch.empty(0, *x_megabatch.shape[1:], device=x_megabatch.device, dtype=x_megabatch.dtype)

        for i in range(self.num_models):
            start_idx = i * batch_size_per_model
            end_idx = (i + 1) * batch_size_per_model
            # Ensure indices are within bounds, especially if x_megabatch is smaller than expected
            # This could happen if some tasks didn't contribute to the batch.
            # However, our construction ensures x_megabatch length is num_active_contributing_models * batch_size_per_model
            if start_idx >= x_megabatch.shape[0]:
                break # No more data in megabatch for further models
            
            sub_batch = x_megabatch[start_idx:min(end_idx, x_megabatch.shape[0])] # min to avoid overrun
            
            if sub_batch.shape[0] > 0:
                 outputs.append(self.models[i](sub_batch))
        
        if not outputs:
            return torch.empty(0, *x_megabatch.shape[1:], device=x_megabatch.device, dtype=x_megabatch.dtype)
            
        return torch.cat(outputs, dim=0)

# --- 4. Helper Functions ---

def create_array_from_grid(
    small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int, hidden_channels_nca_state: int
) -> np.ndarray:
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    # Initialize color channel 0 (empty) to 1.0 for all cells
    arr[:, :, 0] = 1.0
    # Hidden channels (from n_classes to in_channels-1) are already 0.0 by np.zeros

    small_grid_np = np.array(small_grid, dtype=np.int32)
    rows, cols = small_grid_np.shape
    max_rows, max_cols = min(rows, grid_size), min(cols, grid_size)

    for i in range(max_rows):
        for j in range(max_cols):
            pixel_val = small_grid_np[i, j]
            # ARC colors 0-9 map to one-hot channels 1-10 (if n_classes is 11)
            if 0 <= pixel_val <= (n_classes - 2): # Max valid color value
                arr[i, j, :n_classes] = 0.0       # Reset all color channels for this pixel
                arr[i, j, pixel_val + 1] = 1.0    # Set the specific color channel
            # Else (pixel_val is outside 0 to n_classes-2), it remains "empty" (channel 0 is 1.0)
    return arr

def tensor_to_grid(state_tensor: torch.Tensor, n_classes: int) -> List[List[int]]:
    # state_tensor: [C, H, W]
    # Take argmax over the n_classes color channels
    pred_indices = state_tensor.cpu()[:n_classes, :, :].argmax(dim=0).numpy()
    # Map one-hot channel index back to ARC color value:
    # Channel 0 -> color -1 (will be handled by depad or final cleanup)
    # Channel 1 -> color 0
    # ...
    # Channel 10 -> color 9 (if n_classes is 11)
    grid = (pred_indices - 1).tolist() 
    return grid

def depad_grid(grid: List[List[int]], padding_value: int = -1) -> List[List[int]]:
    if not grid or not grid[0]: return [[0]] 

    rows = len(grid)
    cols = len(grid[0])
    min_r, max_r, min_c, max_c = rows, -1, cols, -1 
    found_non_padding = False

    for r_idx in range(rows):
        for c_idx in range(cols):
            if grid[r_idx][c_idx] != padding_value:
                if not found_non_padding: min_r = r_idx # First non-padding row
                max_r = max(max_r, r_idx)
                min_c = min(min_c, c_idx)
                max_c = max(max_c, c_idx)
                found_non_padding = True
    
    if not found_non_padding: return [[0]]

    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]

# --- 5. The "Workhorse" Function for Multi-NCA ---

def run_training_and_prediction_multi_nca(
    all_tasks_data: Dict[str, Dict], 
    active_task_ids: List[str],     
    task_id_to_model_idx: Dict[str, int], # Not strictly needed if active_task_ids maintains order
    device: torch.device,
    hparams: Dict[str, Any]
) -> Dict[str, List[List[List[int]]]]:

    grid_init_args = (
        hparams['grid_size'], 
        hparams['in_channels'], 
        hparams['n_classes'],
        hparams['hidden_channels_nca_state']
    )
    prepared_train_data = {} 

    for task_id in active_task_ids: # active_task_ids should already be filtered
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

    multi_nca_model = MultiCellularNN(
        num_models=num_active_models,
        in_channels=hparams['in_channels'],
        n_classes=hparams['n_classes'],
        nn_hidden_dim=hparams['nn_hidden_dim']
    ).to(device)

    optimizer = torch.optim.Adam(
        multi_nca_model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay']
    )

    batch_size_per_task = hparams['batch_size_per_task']

    print(f"Starting training for {num_active_models} tasks with {hparams['num_iterations']} iterations.")
    multi_nca_model.train()
    for i in range(hparams['num_iterations']):
        mega_batch_inputs_list = []
        mega_batch_targets_list = []
        
        num_tasks_contributing_to_batch = 0

        # Iterate through active_task_ids, their order corresponds to model indices
        for task_id in active_task_ids: 
            task_train_inputs = prepared_train_data[task_id]['inputs']
            task_train_targets = prepared_train_data[task_id]['targets']
            num_samples_for_task = len(task_train_inputs)

            if num_samples_for_task == 0 or batch_size_per_task == 0:
                continue 

            batch_indices = np.random.choice(
                num_samples_for_task, 
                batch_size_per_task, 
                replace=num_samples_for_task < batch_size_per_task # Sample with replacement if needed
            )
            
            mega_batch_inputs_list.extend([task_train_inputs[j] for j in batch_indices])
            mega_batch_targets_list.extend([task_train_targets[j] for j in batch_indices])
            num_tasks_contributing_to_batch +=1


        if not mega_batch_inputs_list:
            if i % 200 == 0: print(f"  Iter {i:04d}/{hparams['num_iterations']}: No data sampled for this batch.")
            continue

        inp_megabatch = torch.stack(mega_batch_inputs_list).to(device)
        target_megabatch = torch.stack(mega_batch_targets_list).to(device)
        
        optimizer.zero_grad()

        state_megabatch = inp_megabatch
        num_nca_steps = np.random.randint(hparams['train_steps_min'], hparams['train_steps_max'] + 1)
        for _ in range(num_nca_steps):
            state_megabatch = multi_nca_model(state_megabatch, batch_size_per_model=batch_size_per_task) 
        
        target_labels_megabatch = target_megabatch[:, :hparams['n_classes']].argmax(dim=1)
        loss = F.cross_entropy(state_megabatch[:, :hparams['n_classes']], target_labels_megabatch)
        
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f"  Iter {i:04d}/{hparams['num_iterations']}: Mega-Loss={loss.item():.4f} (from {num_tasks_contributing_to_batch} tasks)")
            
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"multi_nca_model_{timestamp}.pth")
    torch.save(multi_nca_model.state_dict(), final_checkpoint_path)
    print(f"  Saved final multi-NCA model to {final_checkpoint_path}")

    print("\nStarting prediction phase...")
    multi_nca_model.eval()
    all_task_predictions = {} 

    with torch.no_grad():
        for original_task_id_idx, original_task_id in enumerate(all_tasks_data.keys()):
            task_content = all_tasks_data[original_task_id]
            test_inputs_for_task = task_content['test']
            task_predicted_grids = []
            
            if original_task_id not in active_task_ids:
                # print(f"  Task {original_task_id}: No model trained. Predicting defaults.")
                task_predicted_grids = [[[0]] for _ in test_inputs_for_task]
            else:
                # Find the model_idx corresponding to this original_task_id
                # This relies on active_task_ids being the source of truth for model order
                try:
                    model_idx = active_task_ids.index(original_task_id)
                except ValueError:
                    # Should not happen if active_task_ids is derived correctly from all_tasks_data
                    print(f"Error: Active task {original_task_id} not found in active_task_ids list during prediction.")
                    task_predicted_grids = [[[0]] for _ in test_inputs_for_task] # Default on error
                    all_task_predictions[original_task_id] = task_predicted_grids
                    continue

                specific_model = multi_nca_model.models[model_idx]
                if original_task_id_idx % 50 == 0 : # Print progress occasionally
                     print(f"  Predicting for task {original_task_id} (model {model_idx+1}/{num_active_models})")
                
                for test_case in test_inputs_for_task:
                    test_input_grid = test_case['input']
                    inp_array = create_array_from_grid(test_input_grid, *grid_init_args)
                    inp_tensor = torch.tensor(inp_array).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    current_state = inp_tensor
                    for _ in range(hparams['prediction_steps']):
                        current_state = specific_model(current_state)

                    grid_from_tensor = tensor_to_grid(current_state.squeeze(0), hparams['n_classes'])
                    depadded_grid = depad_grid(grid_from_tensor, padding_value=-1) 
                    final_output_grid = [[max(0, cell) for cell in row] for row in depadded_grid]
                    task_predicted_grids.append(final_output_grid)
            
            all_task_predictions[original_task_id] = task_predicted_grids
            
    return all_task_predictions

# --- 6. Main Execution Block ---

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

    # active_task_ids determines which models are created and their order in MultiCellularNN
    # Sorting ensures consistent model order if tasks are re-read (e.g. for loading checkpoints)
    active_task_ids = sorted([ 
        task_id for task_id, data in all_challenges_data.items() 
        if data.get('train') and len(data['train']) > 0 
    ])
    
    if not active_task_ids:
        print("No tasks with training data found. Generating default submission for all tasks.")
        for task_id_key in all_task_ids_from_file:
            num_test_cases = len(all_challenges_data[task_id_key]['test'])
            submission[task_id_key] = [
                {"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(num_test_cases)
            ]
    else:
        print(f"Identified {len(active_task_ids)} active tasks for parallel training: {active_task_ids[:5]}... (first 5)")
        # task_id_to_model_idx is implicitly handled by the order of active_task_ids
        task_predictions_map = run_training_and_prediction_multi_nca(
            all_tasks_data=all_challenges_data, # Pass all data for access to 'test' fields
            active_task_ids=active_task_ids,     # Only these are trained
            task_id_to_model_idx=None,           # Not strictly needed due to ordered active_task_ids
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