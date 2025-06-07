import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

# --- All the code the worker needs is self-contained here ---

# 1. The CellularNN Model (Copied directly)
class CellularNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, nn_hidden_dim: int):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
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
        color_state = x[:, :self.n_classes]
        padded_color_state = F.pad(color_state, (1, 1, 1, 1), mode='constant', value=0.0)
        B, C_state, H, W = color_state.shape
        neighbor_tensors = []
        for r_offset in range(-1, 2):
            for c_offset in range(-1, 2):
                if r_offset == 0 and c_offset == 0: continue
                neighbor_slice = padded_color_state[:, :, 1+r_offset:H+1+r_offset, 1+c_offset:W+1+c_offset]
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

# 2. Helper Functions (Copied directly)
def create_array_from_grid(small_grid: List[List[int]], grid_size: int, in_channels: int, n_classes: int) -> np.ndarray:
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    arr[:, :, 0] = 1.0
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


# 3. The Workhorse Function (Copied directly, but we don't need CHECKPOINT_DIR here)
def train_and_predict_for_task(
    task_id: str,
    train_pairs: List[Dict],
    test_inputs: List[Dict],
    device: torch.device,
    hparams: Dict[str, Any]
) -> Tuple[str, List[List[int]]]:
    # This function is now fully self-contained with its dependencies above
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
        loss = F.cross_entropy(state[:, :hparams['n_classes']], target_labels)
        loss.backward()
        optimizer.step()

    # NOTE: We can't save checkpoints from here easily without passing the path.
    # For now, we'll skip saving checkpoints from the parallel worker.

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
    """
    A simple wrapper to unpack arguments for the pool and call the main function.
    """
    task_id, train_pairs, test_inputs, device, hparams = args
    
    if not train_pairs:
        predicted_grids = [[[0]] for _ in test_inputs]
        return (task_id, predicted_grids)
    else:
        try:
            return train_and_predict_for_task(
                task_id, train_pairs, test_inputs, device, hparams
            )
        except Exception as e:
            # It's good practice to print the specific task that failed.
            print(f"!!! ERROR processing task {task_id}: {e}", flush=True)
            predicted_grids = [[[0]] for _ in test_inputs]
            return (task_id, predicted_grids)