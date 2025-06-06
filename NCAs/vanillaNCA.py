# %% Imports and Setup Functions
import os
import json
import random
import sys
import csv
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def setup_environment(config: Dict[str, Any]) -> torch.device:
    """Sets up the environment, handling Colab mounting and device selection."""
    if os.getenv('KAGGLE_KERNEL_RUN_TYPE'):
        print("Kaggle environment detected.")
        config['environment']['is_kaggle'] = True
        config['environment']['gColab'] = False # Ensure gColab is false if on Kaggle
    else:
        config['environment']['is_kaggle'] = False # Explicitly set if not Kaggle
    if config['environment']['gColab']:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted.")
        except ImportError:
            print("Google Colab environment detected but 'drive' import failed.")
            config['environment']['gColab'] = False # Fallback if mount fails
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
            config['environment']['gColab'] = False

    if torch.backends.mps.is_available() and config['environment']['allow_mps']:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

# %% Central Configuration
# ==============================================================================
#                           CONFIGURATION SETTINGS
# ==============================================================================
CONFIG: Dict[str, Any] = {
    # --- Core Settings ---
    "run_mode": "train",             # 'train' or 'inference'
    "processing_scope": "batch_tasks", # 'single_task' or 'batch_tasks'

    # --- Task Identification (used based on processing_scope) ---
    "single_task": {
        "json_id": "15663ba9",         # ARC task ID (used if scope is 'single_task')
        "arc_task_subdir": "training", # Subdirectory within dataset_base_dir for the task json
    },
    "batch_tasks": {
        # DIRECTORY containing the .json task files to process
        "tasks_directory": "dataset/ARC-1/data/training", # Relative to script_base or colab_base
        # Limit the number of tasks processed? Set to None to process all found.
        "max_tasks_to_process": None, # Example: 50
        # Limit pairs per task? Set to None to load all pairs.
        "max_pairs_per_task": None, # Example: 10
    },
    "environment": {
        "gColab": True,               # Set to True if running on Google Colab
        "allow_mps": True,             # Allow using Apple Metal Performance Shaders if available
        "is_kaggle": False,
    },

    "paths": {
        "script_base": ".",
        "colab_base": "/content/drive/MyDrive/arcNCAs", # Example Colab path
        "kaggle_input_dir": "/kaggle/input/arc-prize-2025", # Base for Kaggle inputs
        "kaggle_output_dir": "/kaggle/working",            # Base for Kaggle outputs
        "dataset_base_dir_name": "dataset/data", # Name of base dataset dir
        "results_base_subdir": "benchmarking-train-v1",         # Base subdir for all run outputs
        "run_name_prefix": "run_",            # Base prefix for single task run folders
        "batch_run_name": "run_all",         # Specific name for batch task run folder
        # Specific output directories (checkpoints, logs, viz) will be created inside the run folder(s)
    },

    "data": {
        "grid_size": 30,
        "nbd_pad": 1,                  # Neighborhood padding size (1 for 3x3)
        "augment_train": False,        # Apply color permutation augmentation to training data
        "clean_slate_if_white": False, # Reset hidden state if cell predicts class 0 (white/empty)
    },

    "model": {
        # Model architecture is shared across tasks
        "in_channels": 20,             # Total channels (state + hidden)
        "n_classes": 11,               # Number of output classes (0-9 colors + empty)
        "nn_hidden_dim": 128,          # Hidden dimension inside the 1x1 conv update rule
        # Derived values ('hidden_channels', 'perception_channels') calculated later
    },

    "training": {
        "continue_previous_training": False, # Resume from latest matching run?
        # These settings apply PER TASK when training in batch_tasks mode
        "batch_size": 10,               # Number of examples per batch update within a task
        "num_iterations": 1000,       # Total number of BATCH updates PER TASK
        "lr": 1e-3,                    # Learning rate
        "weight_decay": 1e-4,          # Weight decay (L2 regularization on weights)
        "max_norm": 1.0,              # Gradient clipping max norm
        "fire_rate": 1.0,              # Update probability (1.0 for deterministic training steps)
        "min_steps": 30,               # Min steps per training sequence unroll
        "max_steps": 30,               # Max steps per training sequence unroll
        "checkpoint_interval": 1000,    # Save checkpoint every N iterations within a task
        "test_vis_interval": 5000,      # Run test visualization every N iterations within a task
        "use_threading": False,
        "num_workers": 0,              # DataLoader workers
    },

    "inference": {
        # These settings apply PER TASK when running inference
        "batch_size": 1,               # Usually 1 for inference, esp. with task-specific weights
        "update_steps": 30,            # Number of steps for final inference per task
        "fire_rate": 1.0,              # Typically deterministic for inference
        "custom_checkpoint_path": None,# Optional: FULL path to a specific checkpoint file (for single_task mode)
        "output_subdir_per_task": "inference_outputs", # Subdir name within each task's result folder
    },

    "visualization": {
        "colors": ["white", "black", "blue", "red", "green", "yellow", "grey", "pink", "orange", "cyan", "darkred"], # Ensure enough colors for n_classes+
        "figsize": (4, 4),
        "test_vis_subdir_per_task": "test_outputs", # Subdir name within task's folder for test viz during training
        "train_vis_subdir_per_task": "train_outputs",# Subdir name within task's folder for train viz during training
    }
}

if CONFIG['run_mode'] == 'train':
    matplotlib.use('Agg')

# --- Derived Configuration Values ---
def update_derived_config(config: Dict[str, Any], resume_run_path: Optional[Path] = None) -> Dict[str, Any]:
    """Calculates and adds derived configuration values based on the base config."""
    cfg = config.copy() # Avoid modifying the original dict directly

    # --- Model derived values ---
    cfg['model']['hidden_channels'] = cfg['model']['in_channels'] - cfg['model']['n_classes']
    nbd_size = (2 * cfg['data']['nbd_pad'] + 1)**2
    perception_neighbors = nbd_size - 1
    cfg['model']['perception_channels'] = cfg['model']['in_channels'] + (perception_neighbors * cfg['model']['n_classes'])
    if cfg['model']['perception_channels'] <= 0 or cfg['model']['hidden_channels'] < 0:
        raise ValueError("Invalid model channel configuration derived.")

    # --- Path setup based on environment ---
    is_colab = cfg['environment']['gColab']
    is_kaggle = cfg['environment'].get('is_kaggle', False)

    if is_colab:
        active_base_path_for_outputs = Path(cfg['paths']['colab_base'])
        active_base_path_for_inputs = Path(cfg['paths']['colab_base'])
        # For Colab, dataset_base_dir is colab_base / dataset_base_dir_name
        # tasks_directory and arc_task_subdir are relative to this.
        cfg['paths']['dataset_base_dir'] = active_base_path_for_inputs / cfg['paths']['dataset_base_dir_name']
        # batch_tasks_input_dir becomes colab_base / original_tasks_directory_value
        cfg['paths']['batch_tasks_input_dir'] = active_base_path_for_inputs / cfg['batch_tasks']['tasks_directory']
        # single_task.arc_task_subdir remains relative to dataset_base_dir

    elif is_kaggle:
        active_base_path_for_outputs = Path(cfg['paths']['kaggle_output_dir'])
        active_base_path_for_inputs = Path(cfg['paths']['kaggle_input_dir'])
        # --- Kaggle-specific path adjustments ---
        # For Kaggle, 'tasks_directory' (for batch_tasks) and 'arc_task_subdir' (for single_task)
        # must resolve to specific .json filenames (e.g., "arc-agi_training_challenges.json").
        # The initial CONFIG might have placeholder directory-like values for these.
        # This logic maps them to the standard Kaggle filenames if they aren't already .json files.
        # This also respects overrides if the calling script (e.g., main() for a second pass)
        # has already set these to specific .json filenames.

        default_training_filename = "arc-agi_test_challenges.json"
        default_inference_filename = "arc-agi_test_challenges.json" # Standard name for the test file provided by Kaggle

        # Adjust 'batch_tasks.tasks_directory' for Kaggle
        current_batch_tasks_dir_val = cfg['batch_tasks']['tasks_directory']
        if not str(current_batch_tasks_dir_val).endswith(".json"):
            if cfg['run_mode'] == 'train':
                cfg['batch_tasks']['tasks_directory'] = default_training_filename
            elif cfg['run_mode'] == 'inference':
                cfg['batch_tasks']['tasks_directory'] = default_inference_filename
            else:
                raise ValueError(f"Unexpected run_mode '{cfg['run_mode']}' for Kaggle path configuration.")
        # Now cfg['batch_tasks']['tasks_directory'] should be the correct .json filename for Kaggle for this pass.
        cfg['paths']['batch_tasks_input_dir'] = active_base_path_for_inputs / cfg['batch_tasks']['tasks_directory']
        # Adjust 'single_task.arc_task_subdir' for Kaggle
        current_single_task_subdir_val = cfg['single_task']['arc_task_subdir']
        if not str(current_single_task_subdir_val).endswith(".json"):
            if cfg['run_mode'] == 'train':
                cfg['single_task']['arc_task_subdir'] = default_training_filename
            elif cfg['run_mode'] == 'inference':
                cfg['single_task']['arc_task_subdir'] = default_inference_filename
            else:
                raise ValueError(f"Unexpected run_mode '{cfg['run_mode']}' for Kaggle path configuration.")
        # Now cfg['single_task']['arc_task_subdir'] should be the correct .json filename for Kaggle.

        # For single_task on Kaggle, dataset_base_dir is the root input directory (e.g., /kaggle/input/arc-prize-2025).xf
        cfg['paths']['dataset_base_dir'] = active_base_path_for_inputs
    else: # Local execution (not Colab, not Kaggle)
        active_base_path_for_outputs = Path(cfg['paths']['script_base'])
        active_base_path_for_inputs = Path(cfg['paths']['script_base'])
        # Local behavior similar to Colab, paths relative to script_base / dataset_base_dir_name
        cfg['paths']['dataset_base_dir'] = active_base_path_for_inputs / cfg['paths']['dataset_base_dir_name']
        cfg['paths']['batch_tasks_input_dir'] = active_base_path_for_inputs / cfg['batch_tasks']['tasks_directory']

    # Store the effective output directory for general use (e.g. submission.json)
    cfg['paths']['effective_output_dir'] = active_base_path_for_outputs

    # --- Run Name and Main Results Directory (Conditional based on resume_run_path) ---
    scope = cfg['processing_scope']
    # Results are always relative to the determined active_base_path_for_outputs
    results_base_subdir_path = active_base_path_for_outputs / cfg['paths']['results_base_subdir']
    print(scope)
    # Determine the base name prefix before timestamp/resumption logic
    if scope == 'single_task':
        print("1")
        task_id = cfg['single_task']['json_id']
        base_run_name = f"{cfg['paths']['run_name_prefix']}single_{cfg['single_task']['arc_task_subdir']}_{task_id}" # Include subdir in name
    elif scope == 'batch_tasks':
        print("2")
        base_run_name = f"{cfg['paths']['batch_run_name']}"
    else:
        print("3")
        raise ValueError(f"Invalid processing_scope: {scope}")

    if resume_run_path is not None and cfg['run_mode'] == 'train':
        # Resuming: Use the provided path and skip generating a new name/timestamp
        print(f"Resuming training: Using existing directory -> {resume_run_path}")
        cfg['paths']['main_results_dir'] = resume_run_path
    else:
        # Starting fresh or inference: Generate timestamped name
        run_timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        run_folder_name = f"{base_run_name}_{run_timestamp}"
        cfg['paths']['main_results_dir'] = results_base_subdir_path / run_folder_name


    if scope == 'single_task':
        task_id = cfg['single_task']['json_id']
        # Define the single task JSON path, which depends on the environment.
        # is_kaggle, is_colab flags are set earlier in this function.
        if is_kaggle:
            # On Kaggle:
            # - cfg['paths']['dataset_base_dir'] is the root input dir (e.g., /kaggle/input/arc-prize-2025).
            # - cfg['single_task']['arc_task_subdir'] has been updated to be the .json filename (e.g., "arc-agi_training-challenges.json").
            cfg['paths']['task_json_path'] = cfg['paths']['dataset_base_dir'] / cfg['single_task']['arc_task_subdir']
        else: # Local or Colab execution
            # - cfg['paths']['dataset_base_dir'] is <base_path>/dataset_base_dir_name (e.g., ./dataset/data).
            # - cfg['single_task']['arc_task_subdir'] is a subdirectory name (e.g., "training3").
            cfg['paths']['task_json_path'] = (cfg['paths']['dataset_base_dir'] /
                                              cfg['single_task']['arc_task_subdir'] / f"{task_id}.json")
        # For single task, task-specific paths ARE the main paths
        cfg['paths']['task_results_dir'] = cfg['paths']['main_results_dir'] # Checkpoints etc. go here
        cfg['paths']['task_checkpoint_dir'] = cfg['paths']['task_results_dir'] / "checkpoints"
        cfg['paths']['task_log_path'] = cfg['paths']['task_results_dir'] / f"log_{task_id}.txt"
        cfg['paths']['task_loss_csv_path'] = cfg['paths']['task_results_dir'] / f"loss_{task_id}.csv"
        cfg['paths']['task_loss_curve_path'] = cfg['paths']['task_results_dir'] / f"loss_{task_id}.png"
        cfg['paths']['task_config_save_path'] = cfg['paths']['task_results_dir'] / f"config_{task_id}.json"
        cfg['paths']['task_train_vis_dir'] = cfg['paths']['task_results_dir'] / cfg['visualization']['train_vis_subdir_per_task']
        cfg['paths']['task_test_vis_dir'] = cfg['paths']['task_results_dir'] / cfg['visualization']['test_vis_subdir_per_task']
        cfg['paths']['task_inference_output_dir'] = cfg['paths']['task_results_dir'] / cfg['inference']['output_subdir_per_task']

    elif scope == 'batch_tasks':
        # cfg['paths']['batch_tasks_input_dir'] is already set above based on environment
        print("")
        # Main results dir already set above (either resumed or new timestamped)
        # Task-specific subdirectories (checkpoints, logs etc.) will be created INSIDE this main batch run folder during execution
    else:
        raise ValueError(f"Invalid processing_scope: {scope}")

    # --- Visualization cmap ---
    cfg['visualization']['cmap'] = ListedColormap(cfg['visualization']['colors'][:cfg['model']['n_classes']])

    return cfg

# ==============================================================================
#                      DATA HANDLING & TASK LOADING
# ==============================================================================

# create_array_from_grid remains the same as before
def create_array_from_grid(small_grid: List[List[int]], config: Dict[str, Any]) -> np.ndarray:
    """Creates a (grid_size, grid_size, in_channels) numpy array from a small grid."""
    grid_size = config['data']['grid_size']
    in_channels = config['model']['in_channels']
    n_classes = config['model']['n_classes']
    arr = np.zeros((grid_size, grid_size, in_channels), dtype=np.float32)
    arr[:, :, 0] = 1.0 # Initialize as "no pixel" (class 0)

    try:
        small_grid_np = np.array(small_grid, dtype=np.int32)
        rows, cols = small_grid_np.shape
    except ValueError as e:
        print(f"Warning: Could not convert grid to numpy array: {small_grid}. Error: {e}. Skipping grid.")
        return arr # Return empty grid

    max_rows, max_cols = min(rows, grid_size), min(cols, grid_size)
    for i in range(max_rows):
        for j in range(max_cols):
            pixel_val = small_grid_np[i, j]
            if 0 <= pixel_val <= (n_classes - 2): # Valid color range (0-9 for n_classes=11)
                arr[i, j, :n_classes] = 0.0      # Reset class channels
                arr[i, j, pixel_val + 1] = 1.0   # Set the specific class channel
            else:
                arr[i, j, :n_classes] = 0.0      # Keep as class 0 (empty)
                arr[i, j, 0] = 1.0
    return arr

# _apply_augmentation remains the same
def _apply_augmentation(train_pairs: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """Applies color permutation augmentation to a list of training pairs."""
    if not config['data']['augment_train'] or not train_pairs:
        return train_pairs
    print(f"Augmenting {len(train_pairs)} training pairs...")
    # ... (rest of augmentation logic is identical to previous version) ...
    start_time = time.time()
    augmented_train_pairs = []
    processed_count = 0
    for pair in train_pairs:
        augmented_train_pairs.append(pair) # Always include the original pair
        try:
            input_grid = np.array(pair["input"])
            output_grid = np.array(pair["output"])
        except ValueError:
             print(f"Skipping augmentation for invalid pair: {pair}")
             continue
        unique_elements = set(input_grid.flatten()) | set(output_grid.flatten())
        original_colors = sorted([int(x) for x in unique_elements if x != 0])
        if not original_colors: continue
        k = len(original_colors)
        available_colors = list(range(1, 10))
        for p in itertools.permutations(available_colors, k):
            mapping = {original: new for original, new in zip(original_colors, p)}
            mapping[0] = 0
            try:
                vectorized_map = np.vectorize(mapping.get)
                new_input_grid = vectorized_map(input_grid).tolist()
                new_output_grid = vectorized_map(output_grid).tolist()
                augmented_train_pairs.append({"input": new_input_grid, "output": new_output_grid})
            except Exception as e:
                print(f"Error during augmentation mapping for pair {pair}: {e}")
                continue
        processed_count += 1
        if processed_count % 100 == 0: print(f"  Augmented {processed_count}/{len(train_pairs)} original pairs...")
    end_time = time.time()
    print(f"Augmentation complete in {end_time - start_time:.2f} seconds. Total pairs: {len(augmented_train_pairs)}")
    return augmented_train_pairs

# process a single task's data when it's part of a larger Kaggle JSON file
def process_single_kaggle_task_entry(task_id: str, task_content: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Processes a single task's entry from a Kaggle-formatted JSON file.
    'task_content' is the dictionary for a specific task_id, e.g., {"train": [...], "test": [...]}.
    """
    train_pairs = task_content.get("train", [])
    raw_test_items = task_content.get("test", []) # This is a list of "test case" dicts from the JSON task_content
    processed_test_pairs = []

    # Determine the format of "test" items based on the input JSON filename.
    # The 'config' parameter is the main local_config.
    input_json_path_str = ""
    if config['processing_scope'] == 'batch_tasks':
        input_json_path_str = str(config['paths']['batch_tasks_input_dir'])
    elif config['processing_scope'] == 'single_task':
        input_json_path_str = str(config['paths']['task_json_path'])

    # The "arc-agi_test-challenges.json" file has a specific format for its "test" items.
    is_final_submission_file_format = "arc-agi_test-challenges.json" in input_json_path_str

    if is_final_submission_file_format:
        # For "arc-agi_test-challenges.json", each item in "test" is like {"input": grid_data, "output_id": ...}
        # We need to construct pairs like {"input": grid_data} for _process_pairs_to_arrays.
        for test_case_dict in raw_test_items:
            if "input" in test_case_dict: # Ensure "input" key exists
                processed_test_pairs.append({"input": test_case_dict["input"]})
    else:
        # For training/evaluation files (e.g., arc-agi_training-challenges.json),
        # each item in "test" is already a full pair: {"input": ..., "output": ...}.
        # These can be directly used by _process_pairs_to_arrays.
        processed_test_pairs = raw_test_items # raw_test_items is a list of {"input":grid, "output":grid}

    if not train_pairs and not processed_test_pairs:
        print(f"Warning: No train or test pairs found for task_id {task_id} in Kaggle data.")
        # Continue processing as empty data might be valid in some contexts (e.g. submission generation)

    max_pairs = config.get('batch_tasks', {}).get('max_pairs_per_task')
    if max_pairs is not None:
        train_pairs = train_pairs[:max_pairs]
        processed_test_pairs = processed_test_pairs[:max_pairs] # Limit test items as well

    augmented_train_pairs = _apply_augmentation(train_pairs, config)

    train_inputs, train_targets = _process_pairs_to_arrays(augmented_train_pairs, config)
    test_inputs, test_targets = _process_pairs_to_arrays(processed_test_pairs, config) # test_targets will be placeholders or actual targets

    has_train_data = bool(train_inputs)
    if config['run_mode'] == 'train' and not has_train_data:
        print(f"Skipping task {task_id} (from Kaggle data): No training data after processing.")
        return None

    return {
        "task_id": task_id,
        "train_inputs": train_inputs, "train_targets": train_targets,
        "test_inputs": test_inputs, "test_targets": test_targets,
        # "json_path" could point to the main Kaggle file if needed, or be omitted.
    }

# _process_pairs_to_arrays remains the same
def _process_pairs_to_arrays(pairs: List[Dict], config: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Converts a list of grid pairs (dict) into lists of input/target numpy arrays."""
    inputs, targets = [], []
    for pair in pairs:
        try:
            input_arr = create_array_from_grid(pair["input"], config)
            # Handle missing "output" for test data in challenge files
            if "output" in pair:
                target_arr = create_array_from_grid(pair["output"], config)
            else: # For test inputs where output is not given (e.g., from arc-agi_test_challenges.json)
                target_arr = np.zeros((config['data']['grid_size'], config['data']['grid_size'], config['model']['in_channels']), dtype=np.float32)
                target_arr[:, :, 0] = 1.0 # Initialize to "no pixel" / empty state
            inputs.append(input_arr)
            targets.append(target_arr)
        except Exception as e:
            print(f"Warning: Skipping pair due to processing error: {e}. Pair: {pair}")
            continue
    return inputs, targets

# Load data returns task-specific info
def load_task_data(task_json_path: Path, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Loads data for a single task from its JSON file."""
    task_id = task_json_path.stem
    if not task_json_path.exists():
        print(f"Warning: Task JSON file not found: {task_json_path}")
        return None
    try:
        with open(task_json_path, 'r') as f:
            data = json.load(f)
        train_pairs = data.get("train", [])
        test_pairs = data.get("test", [])

        if not train_pairs and not test_pairs:
            print(f"Warning: No train or test pairs found in {task_json_path}")
            # Return info even if empty, caller can decide to skip
            # return None

        # Apply per-task limits if configured (for batch mode loading)
        max_pairs = config.get('batch_tasks', {}).get('max_pairs_per_task')
        if max_pairs is not None:
            train_pairs = train_pairs[:max_pairs]
            test_pairs = test_pairs[:max_pairs]

        # Augment training pairs for this task
        augmented_train_pairs = _apply_augmentation(train_pairs, config)

        # Process pairs into arrays
        train_inputs, train_targets = _process_pairs_to_arrays(augmented_train_pairs, config)
        test_inputs, test_targets = _process_pairs_to_arrays(test_pairs, config)

        # Only return task data if there's something to process for the current mode
        has_train_data = bool(train_inputs)
        has_test_data = bool(test_inputs)

        if (config['run_mode'] == 'train' and not has_train_data):
             print(f"Skipping task {task_id}: No training data after processing.")
             return None
        # Keep task even if test data is missing for inference, maybe infer on train? For now keep it.
        # if (config['run_mode'] == 'inference' and not has_test_data):
        #      print(f"Warning: Task {task_id} has no test data for inference.")


        return {
            "task_id": task_id,
            "train_inputs": train_inputs,
            "train_targets": train_targets,
            "test_inputs": test_inputs,
            "test_targets": test_targets,
            "json_path": task_json_path
        }

    except json.JSONDecodeError as e:
        print(f"Warning: Skipping invalid JSON file: {task_json_path}, Error: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error processing task file {task_json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Modified GridDataset to store task ID
class TaskGridDataset(Dataset):
    """ Custom Dataset storing task ID along with grid pairs. """
    def __init__(self, input_arrays: List[np.ndarray], target_arrays: List[np.ndarray], task_id: str):
        if not isinstance(input_arrays, list) or not isinstance(target_arrays, list):
            raise TypeError("Input and target arrays must be lists.")
        if len(input_arrays) != len(target_arrays):
            raise ValueError(f"Input ({len(input_arrays)}) and target ({len(target_arrays)}) lists differ for task {task_id}")
        if not task_id:
             raise ValueError("Task ID cannot be empty for TaskGridDataset")

        self.input_arrays = input_arrays
        self.target_arrays = target_arrays
        self.task_id = task_id
        if not self.input_arrays:
             print(f"Warning: TaskGridDataset initialized with empty data for task {task_id}.")

    def __len__(self) -> int:
        return len(self.input_arrays)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if not (0 <= idx < len(self.input_arrays)):
             raise IndexError(f"Index {idx} out of range for task {self.task_id} dataset size {len(self.input_arrays)}")

        input_np = self.input_arrays[idx]
        target_np = self.target_arrays[idx]

        try:
            input_tensor = torch.tensor(input_np, dtype=torch.float32).permute(2, 0, 1)
            target_tensor = torch.tensor(target_np, dtype=torch.float32).permute(2, 0, 1)
        except Exception as e:
            raise RuntimeError(f"Tensor conversion failed for index {idx} in task {self.task_id}") from e

        return input_tensor, target_tensor, self.task_id


# Function to create DataLoaders FOR A SINGLE TASK
def create_task_dataloaders(task_data: Dict[str, Any], config: Dict[str, Any], device: torch.device) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Creates training and testing DataLoaders for a specific task's data."""
    train_loader, test_loader = None, None
    pin_memory = device.type != 'cpu'
    task_id = task_data['task_id']

    if task_data['train_inputs']:
        train_dataset = TaskGridDataset(task_data['train_inputs'], task_data['train_targets'], task_id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=pin_memory,
            drop_last=False # Good practice for training stability
        )
    else:
        print(f"No training data for task {task_id}, train_loader not created.")

    if task_data['test_inputs']:
        test_dataset = TaskGridDataset(task_data['test_inputs'], task_data['test_targets'], task_id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['inference']['batch_size'], # Use inference batch size
            shuffle=False, # No shuffle for testing
            num_workers=config['training']['num_workers'],
            pin_memory=pin_memory
        )
    else:
         print(f"No testing data for task {task_id}, test_loader not created.")


    return train_loader, test_loader

# ==============================================================================
#                           MODEL DEFINITION (Identical)
# ==============================================================================
# The CellularNN class remains identical to the previous version.
# It defines the architecture, which is shared across tasks.
# The weights will be loaded/saved specifically for each task.
class CellularNN(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(CellularNN, self).__init__()
        self.model_config = config['model']
        self.data_config = config['data']
        self.in_channels = self.model_config['in_channels']
        self.n_classes = self.model_config['n_classes']
        self.hidden_channels = self.model_config['hidden_channels']
        self.perception_channels = self.model_config['perception_channels']
        self.nn_hidden_dim = self.model_config['nn_hidden_dim']
        self.nbd_pad = self.data_config['nbd_pad']
        self.clean_slate_if_white = self.data_config['clean_slate_if_white']
        self.fire_rate = 1.0 # Default, set externally
        if self.perception_channels <= 0 or self.hidden_channels < 0: raise ValueError("Invalid model channel config.")
        self.fc1 = nn.Conv2d(self.perception_channels, self.nn_hidden_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(self.nn_hidden_dim, self.in_channels, kernel_size=1, bias=True)
        self.layernorm = nn.LayerNorm(self.in_channels)
        self._initialize_weights()
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)
    def set_fire_rate(self, fire_rate: float): self.fire_rate = fire_rate
    def get_neighbor_states(self, x: torch.Tensor) -> torch.Tensor:
        n = self.nbd_pad
        # x: [B, C, H, W], we only care about the first self.n_classes channels
        state = x[:, :self.n_classes]     # [B, C_state, H, W]
        # unfold to [B, C_state * K*K, H*W]
        patches = F.unfold(state, kernel_size=2*n+1, padding=n)
        B, CK2, HW = patches.shape
        K2 = (2*n+1)**2
        # reshape to [B, C_state, K2, H, W]
        patches = patches.view(B, self.n_classes, K2, *state.shape[-2:])
        # drop the center patch
        center = K2 // 2
        neighbor = torch.cat([patches[:,:, :center], patches[:,:, center+1:]], dim=2)
        # flatten back to [B, C_state*(K2-1), H, W]
        return neighbor.reshape(B, self.n_classes*(K2-1), *state.shape[-2:])
    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        neighbor_channels = self.get_neighbor_states(x)
        perception_vector = torch.cat([x, neighbor_channels], dim=1)
        if perception_vector.shape[1] != self.perception_channels: raise ValueError("Perception channel mismatch.")
        return perception_vector
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape; perception = self.perceive(x)
        h = F.relu(self.fc1(perception)); dx = self.fc2(h)
        if self.fire_rate < 1.0:
            update_mask = (torch.rand(B, 1, H, W, device=x.device) <= self.fire_rate).float()
            dx = dx * update_mask
        new_state = x + dx
        if self.clean_slate_if_white:
            with torch.no_grad():
                pred_indices = torch.argmax(new_state[:, :self.n_classes, :, :], dim=1)
                reset_mask = (pred_indices == 0).unsqueeze(1)
                inverse_reset_mask = (~reset_mask).float()
                new_state[:, self.n_classes:] = new_state[:, self.n_classes:] * inverse_reset_mask
        # apply LayerNorm over the C dimension at each spatial location:
        #   permute [B,C,H,W] → [B,H,W,C], normalize last dim, then permute back
        x_perm = new_state.permute(0, 2, 3, 1)       # [B,H,W,C]
        x_norm = self.layernorm(x_perm)             # LN over C
        new_state = x_norm.permute(0, 3, 1, 2)       # [B,C,H,W]
        return new_state

def build_model(config: Dict[str, Any], device: torch.device) -> CellularNN:
    """Builds a new instance of the CellularNN model."""
    # Creates a *new* instance each time it's called.
    print("Building new model instance...")
    model = CellularNN(config).to(device)
    return model

def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
     """Builds the optimizer for a given model instance."""
     lr = config['training']['lr']
     wd = config['training']['weight_decay']
     print(f"Building Adam optimizer with LR={lr}, WeightDecay={wd}")
     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
     return optimizer

# ==============================================================================
#                 VISUALIZATION FUNCTIONS (Mostly Unchanged)
# ==============================================================================
# visualize_state_tensor and visualize_inference_steps_interactive remain identical
# They operate on single tensors or lists of tensors.
def visualize_state_tensor(state_tensor: torch.Tensor, config: Dict[str, Any], filename: Optional[Path] = None):
    """ Visualizes the class predictions from a state tensor (C, H, W) or (B, C, H, W). """
    if state_tensor.dim() == 4:
        if state_tensor.shape[0] > 1: print(f"Visualizing first element of batch size {state_tensor.shape[0]}.")
        elif state_tensor.shape[0] == 0: print("Warning: visualize_state_tensor called with empty batch."); return
        state_tensor = state_tensor[0]
    if state_tensor.dim() != 3: print(f"Warning: visualize_state_tensor expected 3D tensor, got {state_tensor.dim()}D."); return
    state_cpu = state_tensor.detach().cpu(); n_classes = config['model']['n_classes']
    cmap = config['visualization']['cmap']; figsize = config['visualization']['figsize']
    try:
        if state_cpu.shape[0] < n_classes: print(f"Warning: State tensor channels ({state_cpu.shape[0]}) < n_classes ({n_classes})."); return
        logits = state_cpu[:n_classes, :, :]; pred_indices = logits.argmax(dim=0).numpy()
    except Exception as e: print(f"Error during visualization processing: {e}"); return
    plt.figure(figsize=figsize); plt.axis('off')
    plt.imshow(pred_indices, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation='nearest')
    if filename:
        try: filename.parent.mkdir(parents=True, exist_ok=True); plt.savefig(filename, bbox_inches='tight', dpi=150)
        except Exception as e: print(f"Error saving visualization to {filename}: {e}")
        finally: plt.close()
    else: plt.show()

# ==============================================================================
#                CORE TRAINING/INFERENCE (Task-Specific Focus)
# ==============================================================================

# get_target_labels remains the same
def get_target_labels(target_tensor: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """ Extracts target class indices (long type) from the one-hot target tensor. """
    n_classes = config['model']['n_classes']
    target_labels = target_tensor[:, :n_classes, :, :].argmax(dim=1) # [B, H, W]
    return target_labels # Type should be torch.int64

# Modified checkpoint functions to take task-specific paths
def save_task_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, loss: float,
                         task_config: Dict[str, Any]):
    """Saves model and optimizer state for a specific task."""
    chkpt_dir = task_config['paths']['task_checkpoint_dir']
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    chkpt_path = chkpt_dir / f"checkpoint_iter_{iteration:05d}.pth"
    # Config saved is the main config (architecture etc. are shared)
    # We only save the task-specific weights and optimizer state here.
    config_to_save = task_config['main_config'] # Save the original main config
    def config_serializer(obj): # Helper for JSON
        from pathlib import Path
        if isinstance(obj, Path): return str(obj)
        elif isinstance(obj, torch.device): return str(obj)
        try: return json.JSONEncoder().default(obj)
        except TypeError: return f"<{type(obj).__name__} object not serializable>"

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss,
        'config': json.loads(json.dumps(config_to_save, default=config_serializer)) # Save serializable config
    }
    try:
        torch.save(save_dict, chkpt_path)
        print(f"Checkpoint for task {task_config['task_id']} saved to {chkpt_path}")
    except Exception as e:
        print(f"Error saving checkpoint for task {task_config['task_id']} to {chkpt_path}: {e}")


def load_task_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer],
                         task_config: Dict[str, Any], device: torch.device) -> int:
    """Loads the latest checkpoint for a specific task, returns the starting iteration."""
    chkpt_dir = task_config['paths']['task_checkpoint_dir']
    start_iter = 0
    task_id = task_config['task_id']

    if not chkpt_dir.exists():
        print(f"Checkpoint directory not found for task {task_id}: {chkpt_dir}. Starting from scratch.")
        return start_iter

    checkpoints = sorted(chkpt_dir.glob("checkpoint_iter_*.pth"), reverse=True)
    if not checkpoints:
        print(f"No checkpoints found for task {task_id} in {chkpt_dir}. Starting from scratch.")
        return start_iter

    latest_checkpoint = checkpoints[0]
    print(f"Attempting to load checkpoint for task {task_id}: {latest_checkpoint}")
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to correct device
            for state in optimizer.state.values():
                 for k, v in state.items():
                     if isinstance(v, torch.Tensor): state[k] = v.to(device)
        start_iter = checkpoint.get('iteration', 0) + 1
        loaded_loss = checkpoint.get('loss', float('nan'))
        # loaded_config = checkpoint.get('config') # Can verify config if needed
        print(f"Checkpoint loaded for task {task_id}. Resuming from iteration {start_iter}. Last loss: {loaded_loss:.4f}")
    except FileNotFoundError: print(f"Error: Checkpoint file not found: {latest_checkpoint}.")
    except KeyError as e: print(f"Error: Checkpoint for task {task_id} missing key: {e}. Starting scratch."); start_iter = 0
    except Exception as e: print(f"Error loading checkpoint for task {task_id}: {e}. Starting scratch."); start_iter = 0
    return start_iter


def run_task_test_visualization(model: CellularNN, test_loader: DataLoader,
                                task_config: Dict[str, Any], device: torch.device, iteration: int):
    """ Runs inference on the first batch of test data FOR A TASK and saves visualization. """
    if not test_loader:
         # This might happen if a task has no test pairs
         # print(f"Skipping test visualization for task {task_config['task_id']}: No test data loader.")
         return

    model.eval()
    vis_dir = task_config['paths']['task_test_vis_dir']
    vis_dir.mkdir(parents=True, exist_ok=True)
    original_fire_rate = model.fire_rate
    model.set_fire_rate(task_config['main_config']['inference']['fire_rate'])
    n_steps = task_config['main_config']['inference']['update_steps']

    with torch.no_grad():
        try:
            # Dataloader yields (inp, target, task_id) - we ignore task_id here as it's for this task
            inp_batch, target_batch, _ = next(iter(test_loader))
            if inp_batch.shape[0] == 0: print("Warning: Test visualization batch is empty."); return # Safety check

            inp_batch = inp_batch.to(device)
            state = inp_batch.clone()
            for _ in range(n_steps): state = model(state)

            # Visualize the first example of the batch
            filename = vis_dir / f"test_viz_iter_{iteration:05d}.png"
            # Pass the main config for visualization settings (cmap, etc.)
            visualize_state_tensor(state[0], task_config['main_config'], filename=filename)

        except StopIteration: pass # Expected if test_loader is empty
        except Exception as e: print(f"Error during test visualization (task {task_config['task_id']}, iter {iteration}): {e}")
        finally:
            # Restore model state regardless of errors
            model.set_fire_rate(original_fire_rate)
            model.train()

# NEW: Training function focused on a single task
def train_single_task(model: CellularNN, optimizer: optim.Optimizer,
                      train_loader: DataLoader, test_loader: Optional[DataLoader],
                      task_config: Dict[str, Any], device: torch.device):
    """ Training loop for a single task with its own checkpoints and outputs. """

    task_id = task_config['task_id']
    main_config = task_config['main_config'] # Access shared hyperparams
    num_iterations = main_config['training']['num_iterations']
    log_path = task_config['paths']['task_log_path']
    train_vis_dir = task_config['paths']['task_train_vis_dir']
    train_vis_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Checkpoint (Task Specific) ---
    start_iter = load_task_checkpoint(model, optimizer, task_config, device)

    # --- Training Log (Task Specific) ---
    log_mode = 'a' if start_iter > 0 else 'w'
    with open(log_path, log_mode) as log_file:
        if start_iter == 0:
            log_file.write(f"Starting training task {task_id}: {datetime.datetime.now()}\n")
            # Can log task-specific or main config here if desired
            log_file.write("-" * 60 + "\n")
        else:
            log_file.write(f"\nResuming training task {task_id} from iter {start_iter}: {datetime.datetime.now()}\n")
            log_file.write("-" * 60 + "\n")

        # --- Training Loop ---
        losses = []
        model.train()
        model.set_fire_rate(main_config['training']['fire_rate'])

        # Pre‑buffer all training batches once
        all_batches = list(train_loader)            # e.g. one batch of 3–4 examples
        n_batches   = len(all_batches)
        if n_batches == 0:
            print(f"No training batches for task {task_id}, skipping.")
            return

        current_iter = start_iter

        # Extract hyperparams for readability
        n_classes = main_config['model']['n_classes']
        min_steps = main_config['training']['min_steps']
        max_steps = main_config['training']['max_steps']
        max_norm = main_config['training']['max_norm']
        checkpoint_interval = main_config['training']['checkpoint_interval']
        test_vis_interval = main_config['training']['test_vis_interval']

        print(f"Starting training loop for task {task_id} from iter {current_iter} to {num_iterations}")
        loop_start_time = time.time()

        while current_iter < num_iterations:
            # Cycle through buffered batches
            iter_start_time = time.time()
            batch_idx = current_iter % n_batches
            inp_batch, target_batch, task_id_from_dl = all_batches[batch_idx]

            # --- Forward Pass ---
            optimizer.zero_grad()
            inp = inp_batch.to(device)
            target = target_batch.to(device)
            target_labels = get_target_labels(target, main_config)

            n_steps = random.randint(min_steps, max_steps)
            state = inp.clone()
            for step in range(n_steps): state = model(state)

            # --- Loss Calculation ---
            final_logits = state[:, :n_classes, :, :]
            loss = F.cross_entropy(final_logits, target_labels)

            # --- Backward & Step ---
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            # Check grads for NaN/Inf
            found_nan_inf_grad = False
            for param in model.parameters():
                 if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                      print(f"Warning: NaN/Inf grads task {task_id} iter {current_iter}. Skipping step."); log_file.write(f"NaN/Inf grads\n")
                      found_nan_inf_grad = True; break
            if found_nan_inf_grad: optimizer.zero_grad(); current_iter += 1; continue

            optimizer.step()

            # --- Logging & Stats ---
            batch_loss_item = loss.item(); losses.append(batch_loss_item)
            iter_time = time.time() - iter_start_time

            if current_iter % 50 == 0: # Log less frequently
                 log_message = (f"Task {task_id} | Iter {current_iter:05d}/{num_iterations} | Loss: {batch_loss_item:.5f} | "
                                f"GradNorm: {total_norm:.3f} | Steps: {n_steps} | Time: {iter_time:.2f}s")
                 print(log_message); log_file.write(log_message + "\n"); log_file.flush()

            # --- Visualize training state ---
            if current_iter > 0 and current_iter % checkpoint_interval == 0:
                vis_filename = train_vis_dir / f"train_viz_iter_{current_iter:05d}.png"
                visualize_state_tensor(state[0], main_config, filename=vis_filename)

            # --- Save Checkpoint (Task Specific) ---
            if current_iter > 0 and (current_iter % checkpoint_interval == 0 or current_iter == num_iterations - 1):
                 save_task_checkpoint(model, optimizer, current_iter, batch_loss_item, task_config)
                 log_file.write(f"Checkpoint saved at iter {current_iter}\n")

            # --- Run Intermediate Test Visualization (Task Specific) ---
            if test_loader and current_iter > 0 and current_iter % test_vis_interval == 0:
                run_task_test_visualization(model, test_loader, task_config, device, current_iter)
                # Function handles putting model back in train mode

            current_iter += 1

        # --- Post-Loop Actions for Task ---
        total_task_time = time.time() - loop_start_time
        final_msg = f"\nTraining loop for task {task_id} finished after {total_task_time:.2f} seconds."
        print(final_msg); log_file.write(final_msg + "\n")

        if test_loader: # Final test visualization
            print(f"Running final test visualization for task {task_id}...")
            log_file.write("Running final test visualization...\n")
            run_task_test_visualization(model, test_loader, task_config, device, num_iterations)

        # Save/Plot Loss History (Task Specific)
        loss_csv_path = task_config['paths']['task_loss_csv_path']
        loss_curve_path = task_config['paths']['task_loss_curve_path']
        try:
            mode = 'a' if start_iter > 0 else 'w'
            with open(loss_csv_path, mode, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if mode == 'w': writer.writerow(['Iteration', 'Loss'])
                for i, loss_val in enumerate(losses): writer.writerow([start_iter + i, loss_val])
            print(f"Loss history for task {task_id} saved to {loss_csv_path}")
            log_file.write(f"Loss history saved to {loss_csv_path}\n")
            # Plotting
            iterations, all_losses = [], []
            if loss_csv_path.exists():
                 with open(loss_csv_path, 'r') as csvfile:
                      reader = csv.reader(csvfile); next(reader) # Skip header
                      for row in reader:
                           try: iterations.append(int(row[0])); all_losses.append(float(row[1]))
                           except: pass # Skip bad rows
            if all_losses:
                 plt.figure(figsize=(10, 5)); plt.plot(iterations, all_losses)
                 plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title(f"Task {task_id} Training Loss")
                 plt.grid(True); plt.ylim(bottom=0); plt.tight_layout(); plt.savefig(loss_curve_path); plt.close()
                 print(f"Loss curve for task {task_id} saved to {loss_curve_path}")
                 log_file.write(f"Loss curve saved to {loss_curve_path}\n")
        except Exception as e: print(f"Error saving/plotting loss for task {task_id}: {e}")

def tensor_to_grid(state_tensor_single_item: torch.Tensor, config: Dict[str, Any]) -> List[List[int]]:
    """
    Converts a single predicted state tensor [C, H, W] to a List[List[int]] grid.
    The output grid contains class indices (0-10), corresponding to model's direct output.
    It returns the full HxW grid from the tensor. For ARC, this might need to be cropped.
    """
    if state_tensor_single_item.dim() != 3:
        # This can happen if an empty batch or similar unexpected tensor is passed.
        # For robustness in submission generation, return a default small grid.
        print(f"Warning: tensor_to_grid expects a 3D tensor [C,H,W], got {state_tensor_single_item.shape}. Returning default grid.")
        return [[0]] # Default fallback grid, e.g. a single pixel of color 0

    n_classes = config['model']['n_classes']
    # Ensure tensor is on CPU for numpy conversion
    state_tensor_single_item_cpu = state_tensor_single_item.cpu()

    # Get class predictions: argmax over the class dimension (dim 0)
    # These are indices 0 to n_classes-1, assumed to be the direct color values for submission.
    pred_indices_tensor = state_tensor_single_item_cpu[:n_classes, :, :].argmax(dim=0) # Shape [H, W]

    # Convert to list of lists
    grid_h, grid_w = pred_indices_tensor.shape
    output_grid = []
    for r in range(grid_h):
        output_grid.append([int(pred_indices_tensor[r, c].item()) for c in range(grid_w)])
    return output_grid

def depad_grid(grid: List[List[int]], padding_value: int = 0) -> List[List[int]]:
    """
    Removes padding from a grid by finding the smallest bounding box
    containing non-padding values.

    Args:
        grid: The input grid (List[List[int]]).
        padding_value: The integer value used for padding. Defaults to 0.

    Returns:
        The depadded grid (List[List[int]]).
        If the grid is empty or all padding, returns [[padding_value]] (e.g., [[0]]).
    """
    if not grid or not grid[0]:
        return [[padding_value]]

    rows = len(grid)
    cols = len(grid[0])

    min_r, max_r = -1, -1
    true_min_c, true_max_c = cols, -1
    found_non_padding = False

    for r_idx in range(rows):
        current_row_min_c = cols
        current_row_max_c = -1
        row_has_non_padding = False
        for c_idx in range(cols):
            if grid[r_idx][c_idx] != padding_value:
                if not found_non_padding:
                    min_r = r_idx
                max_r = r_idx
                current_row_min_c = min(current_row_min_c, c_idx)
                current_row_max_c = max(current_row_max_c, c_idx)
                row_has_non_padding = True
                found_non_padding = True
        if row_has_non_padding:
            true_min_c = min(true_min_c, current_row_min_c)
            true_max_c = max(true_max_c, current_row_max_c)
    if not found_non_padding:
        return [[padding_value]]
    depadded_grid_result = [row[true_min_c : true_max_c + 1] for row in grid[min_r : max_r + 1]]
    return depadded_grid_result if depadded_grid_result and depadded_grid_result[0] else [[padding_value]]

def train_worker(task_data):
    # Each worker sets its device independently
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = task_data['task_config']['main_config']
    # Create DataLoaders
    train_loader, test_loader = create_task_dataloaders(task_data, cfg, device)
    if not train_loader:
        print(f"Worker skipping {task_data['task_id']}: no train data")
        return task_data['task_id']
    # Build model & optimizer
    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    # Run training
    train_single_task(model, optimizer, train_loader, test_loader,
                      task_data['task_config'], device)
    return task_data['task_id']

# NEW: Inference function for batch mode (loads weights per task)
def run_batch_inference(model: CellularNN, all_tasks_test_loader: DataLoader,
                        main_config: Dict[str, Any], device: torch.device):
    """ Runs inference across multiple tasks, loading the correct weights for each example. """
    print("--- Running Batch Inference (Task-Specific Weights) ---")
    main_results_dir = main_config['paths']['main_results_dir'] # Base dir for the batch run
    inference_subdir_name = main_config['inference']['output_subdir_per_task']
    update_steps = main_config['inference']['update_steps']

    # We need to load checkpoints repeatedly, so no single pre-load
    model.eval()
    model.set_fire_rate(main_config['inference']['fire_rate'])

    current_task_id = None
    loaded_checkpoint_path = None
    example_index = 0 # Global index across all tasks processed

    with torch.no_grad():
        for batch_idx, (inp_batch, target_batch, task_id_batch) in enumerate(all_tasks_test_loader):
            print(f"Processing inference batch {batch_idx + 1}/{len(all_tasks_test_loader)}...")
            inp_batch = inp_batch.to(device)
            target_batch = target_batch.to(device)
            batch_size = inp_batch.shape[0]

            output_states = [] # Store results for the batch

            # Process each item in the batch individually to handle weight loading
            for i in range(batch_size):
                item_task_id = task_id_batch[i]
                item_inp = inp_batch[i:i+1] # Keep batch dim [1, C, H, W]
                item_target = target_batch[i:i+1]
                current_example_idx = example_index + i

                # --- Load Checkpoint for this item's task ID ---
                if item_task_id != current_task_id:
                    print(f"  Switching context to task: {item_task_id}")
                    current_task_id = item_task_id
                    loaded_checkpoint_path = None # Reset loaded path tracker

                    # Construct paths for this task within the main batch run dir
                    # ─── Determine where the checkpoints live ───
                    # For batch‐`mode runs: runs/.../<task_id>/checkpoints
                    # For single‐task runs: runs/.../checkpoints
                    batch_ckpt = (main_results_dir / current_task_id / "checkpoints")
                    single_ckpt = (main_results_dir / "checkpoints")
                    if batch_ckpt.exists():
                        task_chkpt_dir = batch_ckpt
                    elif single_ckpt.exists():
                        task_chkpt_dir = single_ckpt
                    else:
                        # neither layout exists; still point at the batch path so loader will report missing
                        task_chkpt_dir = batch_ckpt

                    task_config_for_load = {
                        'task_id': current_task_id,
                        'paths': {'task_checkpoint_dir': task_chkpt_dir}
                    }

                    # Load latest checkpoint for THIS task (ignore optimizer)
                    # Load into the existing model instance
                    start_iter = load_task_checkpoint(model, None, task_config_for_load, device)

                    if start_iter == 0: # Check if loading failed
                         print(f"  ERROR: Failed to load checkpoint for task {current_task_id}. Skipping example {current_example_idx}.")
                         # Add a placeholder or skip? Let's skip for now.
                         # Need a way to signal this skip to the visualization part.
                         # output_states.append(None) # Append None to signal failure
                         continue # Skip to next item in batch

                    # Update tracker
                    if task_chkpt_dir.exists():
                         checkpoints = sorted(task_chkpt_dir.glob("checkpoint_iter_*.pth"), reverse=True)
                         if checkpoints: loaded_checkpoint_path = checkpoints[0]

                elif loaded_checkpoint_path is None:
                     # This case means we previously failed to load for this task ID
                     print(f"  Skipping example {current_example_idx} for task {current_task_id} (previous load failed).")
                     # output_states.append(None)
                     continue


                # --- Run Inference Steps for this item ---
                state = item_inp.clone()

                for step in range(update_steps):
                    state = model(state)

                output_states.append(state) # Add final state [1, C, H, W]

                # --- Visualize/Save Output (Task-Specific Directory) ---
                task_inference_output_dir = main_results_dir / item_task_id / inference_subdir_name
                task_inference_output_dir.mkdir(parents=True, exist_ok=True)

                final_filename = task_inference_output_dir / f"final_pred_example_{current_example_idx:04d}.png"
                visualize_state_tensor(state, main_config, filename=final_filename) # state is [1, C, H, W]

                target_filename = task_inference_output_dir / f"target_example_{current_example_idx:04d}.png"
                visualize_state_tensor(item_target, main_config, filename=target_filename)

            # End of loop through items in batch
            example_index += batch_size

    print(f"\nBatch inference complete. Outputs saved in task subdirectories under {main_results_dir}")


# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================

def main(initial_config_override: Optional[Dict[str, Any]] = None):
    from pathlib import Path
    main_call_start_time = time.time()

    if initial_config_override:
        # If an override is provided, use it. This is for the second (inference) pass.
        local_config = json.loads(json.dumps(initial_config_override)) # Deep copy
        print(f"\n--- Main called with overridden config, initial run_mode: {local_config.get('run_mode')} ---")
    else:
        # First pass, or if main is called directly without an override (e.g. from global CONFIG)
        local_config = json.loads(json.dumps(CONFIG)) # Deep copy global CONFIG
        print(f"\n--- Main called with global CONFIG, initial run_mode: {local_config.get('run_mode')} ---")

     # --- 1. Setup Environment & Finalize Config ---
    # Store the mode this instance of main() will execute under, for clarity in the calling script
    local_config['run_mode_at_execution'] = local_config['run_mode']

    # --- 1. Setup Environment & Finalize Config ---
    device = setup_environment(local_config)
    local_config['DEVICE'] = device
    # --- Determine if resuming a training run ---
    resume_run_path: Optional[Path] = None
    if local_config['run_mode'] == 'train' and local_config['training']['continue_previous_training']:
        from pathlib import Path
        temp_output_base = Path(local_config['paths']['kaggle_output_dir'] if local_config['environment']['is_kaggle'] else (local_config['paths']['colab_base'] if local_config['environment']['gColab'] else local_config['paths']['script_base']))
        runs_root = temp_output_base / local_config['paths']['results_base_subdir']
        scope = local_config['processing_scope']
        print(f"Continue training requested. Searching for existing runs in: {runs_root}")

        if runs_root.is_dir():
            candidate_dirs = list(runs_root.iterdir())
            candidate_dirs = [d for d in candidate_dirs if d.is_dir()]

            if scope == 'single_task':
                task_id = local_config['single_task']['json_id']
                pattern_start = f"{local_config['paths']['run_name_prefix']}single_{local_config['single_task']['arc_task_subdir']}_{task_id}_"
            elif scope == 'batch_tasks':
                pattern_start = f"{local_config['paths']['batch_run_name']}_"
            else:
                pattern_start = None
                print(f"Warning: Unknown processing scope '{scope}' for resuming.")

            if pattern_start:
                matching_dirs = [d for d in candidate_dirs if d.name.startswith(pattern_start)]
                if matching_dirs:
                    latest_matching_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)
                    resume_run_path = latest_matching_dir
                    print(f"Found latest matching previous run directory: {resume_run_path}")
                else:
                    print(f"No existing run directory found matching pattern '{pattern_start}*'. Starting a new run.")
            else: # No pattern (e.g., bad scope)
                print("Could not determine search pattern. Starting a new run.")
        else:
            print(f"Runs directory '{runs_root}' not found. Starting a new run.")
    try:
        # Run initial config derivation to get base paths, task IDs etc.
        # This will set up paths assuming a *new* run directory.

        # Pass resume_run_path (which is None if not resuming or not in train mode)
        local_config = update_derived_config(local_config, resume_run_path)




        if local_config['run_mode'] == 'inference':
            # This block ONLY runs for inference, not affected by continue_previous_training
            from pathlib import Path
            base = Path(local_config['paths']['kaggle_output_dir'] if local_config['environment']['is_kaggle'] else
                        (local_config['paths']['colab_base'] if local_config['environment']['gColab'] else local_config['paths']['script_base']))
            runs_root = base / local_config['paths']['results_base_subdir']
            found_inference_dir = None # Path to the directory to use for inference

            print(f"Inference mode: Searching for existing run directories in: {runs_root}")

            if runs_root.is_dir():
                candidate_dirs = list(runs_root.iterdir())
                candidate_dirs = [d for d in candidate_dirs if d.is_dir()] # Filter for directories only

                if not candidate_dirs:
                     print("Warning: No existing run directories found.")

                # --- Single Task Inference: Prioritize specific task runs ---
                elif local_config['processing_scope'] == 'single_task':
                    task_id = local_config['single_task']['json_id']
                    pattern_start = f"{local_config['paths']['run_name_prefix']}single_{local_config['single_task']['arc_task_subdir']}_{task_id}_"
                    task_specific_dirs = []
                    for d in candidate_dirs:
                        if d.name.startswith(pattern_start):
                             task_specific_dirs.append(d)

                    if task_specific_dirs:
                        latest_task_dir = max(task_specific_dirs, key=lambda p: p.stat().st_mtime)
                        print(f"Found latest specific run directory for task {task_id}: {latest_task_dir}")
                        found_inference_dir = latest_task_dir
                    else:
                        print(f"Warning: No specific run directory found for task {task_id} (pattern: '{pattern_start}*').")
                        # Fallback: Look for the latest overall directory below.
                        latest_overall_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)
                        print(f"Falling back to latest overall run directory: {latest_overall_dir}")
                        found_inference_dir = latest_overall_dir

                # --- Batch Task Inference: Use the latest overall run ---
                elif local_config['processing_scope'] == 'batch_tasks':
                    batch_run_prefix = local_config['paths']['batch_run_name'] + "_" # e.g., "run_all_"
                    # Filter directories to only include those matching the batch run pattern
                    batch_run_dirs = [d for d in candidate_dirs if d.name.startswith(batch_run_prefix)]

                    if batch_run_dirs:
                        latest_batch_dir = max(batch_run_dirs, key=lambda p: p.stat().st_mtime)
                        print(f"Found latest batch run directory: {latest_batch_dir}")
                        found_inference_dir = latest_batch_dir
                    else:
                        print(f"Warning: No specific batch run directory found (pattern: '{batch_run_prefix}*').")
                        # Fallback: Use the latest overall directory? Or error? For now, warn and use latest overall.
                        # This might lead to errors if the latest overall isn't a compatible batch run structure.
                        if candidate_dirs:
                             latest_overall_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)
                             print(f"Falling back to latest overall run directory: {latest_overall_dir}")
                             found_inference_dir = latest_overall_dir

            else: # runs_root doesn't exist
                 print(f"Warning: Runs directory {runs_root} not found. Will proceed with generated path if needed.")

            # --- Override paths if an existing directory was found ---
            if found_inference_dir:
                print(f"Updating config paths for inference to use found directory: {found_inference_dir}")
                local_config['paths']['main_results_dir'] = found_inference_dir
                # If single task, need to update all derived task paths to point within the found directory
                if local_config['processing_scope'] == 'single_task':
                    task_id = local_config['single_task']['json_id']
                    # These paths were initially set relative to a *new* timestamped directory.
                    # Re-base them relative to the found directory.
                    local_config['paths']['task_results_dir'] = found_inference_dir # For single task, main IS the task dir
                    local_config['paths']['task_checkpoint_dir'] = found_inference_dir / "checkpoints"
                    local_config['paths']['task_log_path'] = found_inference_dir / f"log_{task_id}.txt"
                    local_config['paths']['task_loss_csv_path'] = found_inference_dir / f"loss_{task_id}.csv"
                    local_config['paths']['task_loss_curve_path'] = found_inference_dir / f"loss_{task_id}.png"
                    local_config['paths']['task_config_save_path'] = found_inference_dir / f"config_{task_id}.json" # Check if exists? Less critical for inference
                    local_config['paths']['task_train_vis_dir'] = found_inference_dir / local_config['visualization']['train_vis_subdir_per_task']
                    local_config['paths']['task_test_vis_dir'] = found_inference_dir / local_config['visualization']['test_vis_subdir_per_task']
                    local_config['paths']['task_inference_output_dir'] = found_inference_dir / local_config['inference']['output_subdir_per_task']
            else:
                # No existing directory found or applicable, use the path generated by update_derived_config
                print(f"Using default generated path for inference results: {local_config['paths']['main_results_dir']}")


    except Exception as e:
        import traceback
        print(f"FATAL: Error during configuration setup or directory finding: {e}")
        traceback.print_exc()
        return

    # --- Create Main Run Directory (if it doesn't already exist) ---
    main_results_dir = local_config['paths']['main_results_dir']
    try:
        # Only creates if it doesn't exist. Safe to call even if we found an existing dir.
        main_results_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"FATAL: Cannot create main results dir {main_results_dir}: {e}")
        return
    print(f"Using main results directory: {main_results_dir}")

    # --- Save Main Configuration (Always save to the used results dir) ---
    # Note: If using an existing directory, this might overwrite a previous config.
    # Consider naming it differently for inference runs, e.g., 'config_inference_run.json'
    main_config_save_path = main_results_dir / "config_run_main.json"
    try:
        def config_serializer(obj): # Helper for JSON
            from pathlib import Path
            if isinstance(obj, Path): return str(obj)
            elif isinstance(obj, torch.device): return str(obj)
            try: return json.JSONEncoder().default(obj)
            except TypeError: return f"<{type(obj).__name__} object not serializable>"
        with open(main_config_save_path, "w") as f:
            json.dump(local_config, f, indent=4, default=config_serializer)
        print(f"Effective run configuration saved to: {main_config_save_path}")
    except Exception as e: print(f"Warning: Error saving effective config: {e}")


    # --- 2. Identify and Load Tasks ---
    tasks_to_process: List[Dict[str, Any]] = []
    scope = local_config['processing_scope']

    if scope == 'single_task':
        task_id_to_load = local_config['single_task']['json_id']
        task_json_path = local_config['paths'].get('task_json_path')
        task_data_processed = None

        if not task_json_path or not task_json_path.exists():
            print(f"FATAL: Task JSON path not found or does not exist: {task_json_path}")
            return

        if local_config['environment']['is_kaggle'] and task_json_path.is_file():
            # Kaggle: single_task means loading from a main JSON file and picking one task.
            # `task_json_path` (from `arc_task_subdir` in config) should be the main Kaggle JSON filename.
            print(f"Kaggle single task: Loading task '{task_id_to_load}' from container file: {task_json_path}")
            try:
                with open(task_json_path, 'r') as f:
                    all_tasks_in_container = json.load(f)
                if task_id_to_load not in all_tasks_in_container:
                    print(f"FATAL: Task ID '{task_id_to_load}' not found in container file {task_json_path}. Available tasks: {list(all_tasks_in_container.keys())[:5]}...")
                    return
                task_content = all_tasks_in_container[task_id_to_load]
                task_data_processed = process_single_kaggle_task_entry(task_id_to_load, task_content, local_config)
            except Exception as e:
                print(f"FATAL: Error loading/processing single task '{task_id_to_load}' from container {task_json_path}: {e}")
                return
        else:
            # Original logic: single task from its own dedicated file (e.g., task_id.json in a subdir)
            if task_json_path.is_dir(): # Should point to the file itself, not dir
                 print(f"FATAL: For non-Kaggle single_task, task_json_path should be a file, not directory: {task_json_path}")
                 return
            task_data_processed = load_task_data(task_json_path, local_config)

        if task_data_processed:
            tasks_to_process.append(task_data_processed)
            # Common setup for task_config (applies to both Kaggle-loaded and normally-loaded single tasks)
            task_id_for_config = tasks_to_process[0]['task_id'] # Should match task_id_to_load
            # Ensure paths in local_config['paths'] for 'single_task' are used correctly.
            # These are already set up in update_derived_config for 'single_task' scope.
            tasks_to_process[0]['task_config'] = {
                'task_id': task_id_for_config,
                'main_config': local_config,
                'paths': {
                    'task_results_dir': local_config['paths']['task_results_dir'],
                    'task_checkpoint_dir': local_config['paths']['task_checkpoint_dir'],
                    'task_log_path': local_config['paths']['task_log_path'],
                    'task_loss_csv_path': local_config['paths']['task_loss_csv_path'],
                    'task_loss_curve_path': local_config['paths']['task_loss_curve_path'],
                    'task_config_save_path': local_config['paths']['task_config_save_path'],
                    'task_train_vis_dir': local_config['paths']['task_train_vis_dir'],
                    'task_test_vis_dir': local_config['paths']['task_test_vis_dir'],
                    'task_inference_output_dir': local_config['paths']['task_inference_output_dir']
                }
            }
            Path(tasks_to_process[0]['task_config']['paths']['task_inference_output_dir']).mkdir(parents=True, exist_ok=True)

    elif scope == 'batch_tasks':
        # This path is set by update_derived_config.
        # For Kaggle, if tasks_directory in CONFIG is a filename, this will point to the file.
        # For non-Kaggle, it points to a directory.
        tasks_input_path = local_config['paths']['batch_tasks_input_dir']

        if not tasks_input_path.exists():
            print(f"FATAL: Batch tasks input path not found: {tasks_input_path}")
            return

        if local_config['environment']['is_kaggle'] and tasks_input_path.is_file():
            # Kaggle batch mode: Load tasks from a single main JSON file.
            print(f"Kaggle batch mode: Loading tasks from single file: {tasks_input_path}")
            try:
                with open(tasks_input_path, 'r') as f:
                    all_kaggle_tasks_data = json.load(f) # This is a dict: {task_id: task_content}
            except Exception as e:
                print(f"FATAL: Could not load Kaggle task file {tasks_input_path}: {e}")
                return

            max_tasks = local_config['batch_tasks']['max_tasks_to_process']
            loaded_count = 0
            for task_id, task_content in all_kaggle_tasks_data.items():
                if max_tasks is not None and loaded_count >= max_tasks:
                    print(f"Reached max tasks limit ({max_tasks}).")
                    break
                print(f"  Processing Kaggle task_id: {task_id}...")
                task_data_processed = process_single_kaggle_task_entry(task_id, task_content, local_config)
                if task_data_processed:
                    tasks_to_process.append(task_data_processed)
                    loaded_count += 1
                else:
                    print(f"  Skipped Kaggle task_id: {task_id} (processing error or no valid data).")

        elif tasks_input_path.is_dir():
            # Original logic: load individual .json files from a directory
            print(f"Scanning for tasks in directory: {tasks_input_path}")
            json_files = sorted(list(tasks_input_path.glob("*.json")))
            print(f"Found {len(json_files)} potential task files.")

            max_tasks = local_config['batch_tasks']['max_tasks_to_process']
            loaded_count = 0
            for json_path in json_files:
                if max_tasks is not None and loaded_count >= max_tasks:
                    print(f"Reached max tasks limit ({max_tasks}).")
                    break
                print(f"  Loading task: {json_path.name}...")
                task_data_processed = load_task_data(json_path, local_config) # Original function
                if task_data_processed:
                    tasks_to_process.append(task_data_processed)
                    loaded_count += 1
                else:
                    print(f"  Skipped task: {json_path.name} (no valid data or error)")
        else:
            print(f"FATAL: Batch tasks input path is not a recognized file or directory: {tasks_input_path}")
            return

        # Common processing for all loaded tasks in batch mode (Kaggle or not)
        for i in range(len(tasks_to_process)):
            task_data = tasks_to_process[i]
            task_id = task_data['task_id']
            task_results_dir = main_results_dir / task_id # Task-specific results dir
            task_data['task_config'] = {
                'task_id': task_id,
                'main_config': local_config,
                'paths': {
                    'task_results_dir': task_results_dir,
                    'task_checkpoint_dir': task_results_dir / "checkpoints",
                    'task_log_path': task_results_dir / f"log_{task_id}.txt",
                    'task_loss_csv_path': task_results_dir / f"loss_{task_id}.csv",
                    'task_loss_curve_path': task_results_dir / f"loss_{task_id}.png",
                    'task_config_save_path': task_results_dir / f"config_{task_id}.json",
                    'task_train_vis_dir': task_results_dir / local_config['visualization']['train_vis_subdir_per_task'],
                    'task_test_vis_dir': task_results_dir / local_config['visualization']['test_vis_subdir_per_task'],
                    'task_inference_output_dir': task_results_dir / local_config['inference']['output_subdir_per_task']
                }
            }
            task_results_dir.mkdir(parents=True, exist_ok=True)
            Path(task_data['task_config']['paths']['task_inference_output_dir']).mkdir(parents=True, exist_ok=True)
            # Optionally save individual task configs (can be verbose)
            # try:
            #     with open(task_data['task_config']['paths']['task_config_save_path'], "w") as f_task_cfg:
            #         json.dump(task_data['task_config'], f_task_cfg, indent=4, default=config_serializer)
            # except Exception as e_cfg: print(f"Warning: Could not save task config for {task_id}: {e_cfg}")

    if not tasks_to_process:
        print("FATAL: No tasks loaded or processed successfully. Exiting.")
        return
    print(f"\nSuccessfully loaded {len(tasks_to_process)} tasks for processing.")

    # --- 3. Execute Run Mode ---
    run_mode = local_config['run_mode']

    if run_mode == 'train':
        # ... (training logic remains the same) ...
        print("\n--- Starting Training Run ---")
        use_threads = local_config['training'].get('use_threading', True)

        if use_threads and len(tasks_to_process) > 1 : # Only use threads if multiple tasks
            max_workers = max(1, min(10, os.cpu_count() or 1, len(tasks_to_process))) # Added os.cpu_count()
            print(f"Launching parallel training on {len(tasks_to_process)} tasks using {max_workers} threads...")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(train_worker, tasks_to_process))
        else:
            print(f"Launching sequential training on {len(tasks_to_process)} tasks...")
            results = []
            for td_idx, td in enumerate(tasks_to_process):
                print(f"Starting task {td_idx+1}/{len(tasks_to_process)}: {td['task_id']}")
                results.append(train_worker(td))

        print(f"\nAll training tasks completed: {results}")

    elif run_mode == 'inference':
        print("\n--- Starting Inference Run ---")

        if scope == 'single_task':
            if tasks_to_process:
                task_data = tasks_to_process[0]
                task_id = task_data['task_id']
                task_config_for_load = task_data['task_config'] # This now has correctly scoped paths
                print(f"Running inference for single task: {task_id}")
                print(f"Expecting checkpoints in: {task_config_for_load['paths']['task_checkpoint_dir']}")

                _, test_loader = create_task_dataloaders(task_data, local_config, device)
                if not test_loader:
                    print(f"Cannot run inference for task {task_id}: No test data loaded.")
                else:
                    model = build_model(local_config, device)
                    # For single task, `run_batch_inference` can be used if we wrap the loader
                    # The current `run_batch_inference` expects task_id from the loader.
                    # Let's ensure the TaskGridDataset is used, which provides task_id.
                    # (create_task_dataloaders already uses TaskGridDataset)
                    run_batch_inference(model, test_loader, local_config, device) # test_loader for single task is fine
            else:
                 print("Cannot run inference: Failed to load single task.")

        elif scope == 'batch_tasks':
            print(f"Running serial batch inference for submission generation.")
            print(f"Expecting checkpoints in subdirectories of: {local_config['paths']['main_results_dir']}")

            submission_output_collector = {}
            # Pre-populate submission_output_collector with all task IDs expected in the submission.
            # This relies on the input file for inference containing all tasks to be submitted.
            # For Kaggle, tasks_input_path would be the arc-agi_test_challenges.json
            tasks_input_path_for_submission_keys = local_config['paths']['batch_tasks_input_dir']
            if local_config['environment']['is_kaggle'] and tasks_input_path_for_submission_keys.is_file():
                try:
                    with open(tasks_input_path_for_submission_keys, 'r') as f_keys:
                        submission_task_data = json.load(f_keys)
                    for task_id_key, task_content_key in submission_task_data.items():
                        submission_output_collector[task_id_key] = []
                        num_test_cases_for_task = len(task_content_key.get("test", []))
                        default_prediction_grid = [[0]] # Minimal default grid
                        for _ in range(num_test_cases_for_task):
                            submission_output_collector[task_id_key].append({
                                "attempt_1": default_prediction_grid,
                                "attempt_2": default_prediction_grid  # Kaggle wants up to 2 attempts
                    })
                except Exception as e_keys:
                    print(f"Warning: Could not pre-populate submission keys from {tasks_input_path_for_submission_keys}: {e_keys}")
                    # Fallback: populate as tasks are processed, but this might miss tasks if they fail to load/process.
            else:
                print(f"Warning: Not on Kaggle or input path is not a file ({tasks_input_path_for_submission_keys}). Submission keys might be incomplete if relying on pre-population.")

            model_instance = build_model(local_config, device) # Build one model instance for serial processing

            for task_data in tasks_to_process: # Iterate over successfully loaded tasks
                task_id = task_data['task_id']
                print(f"--- Processing task {task_id} for submission ---")
                if not task_data['test_inputs']:
                    print(f"  Task {task_id} has no test inputs processed. Default predictions (if any expected test cases) will be used.")
                    # Ensure entry exists even if no test inputs were processed by our pipeline (might be in submission file)
                    if task_id not in submission_output_collector:
                        submission_output_collector[task_id] = [] # Will remain empty if no test cases in source file
                    continue # submission_output_collector already has placeholders

                # Load checkpoint for this task
                # Checkpoint path expected by load_task_checkpoint: main_results_dir / task_id / "checkpoints"
                task_chkpt_dir = local_config['paths']['main_results_dir'] / task_id / "checkpoints"
                task_config_for_load = { # Config structure expected by load_task_checkpoint
                    'task_id': task_id,
                    'paths': {'task_checkpoint_dir': task_chkpt_dir},
                    'main_config': local_config # Pass main_config for consistency, though not strictly used by loader
                }
                model_instance.to(device) # Ensure model is on the correct device before loading state_dict
                start_iter = load_task_checkpoint(model_instance, None, task_config_for_load, device)

                if start_iter == 0: # Indicates checkpoint might not have loaded successfully
                    print(f"  Warning: No checkpoint found or failed to load for task {task_id}. Predictions will be from current model state (potentially untrained/random).")
                    # Continue with current model state; predictions will be made.

                model_instance.eval()
                model_instance.set_fire_rate(local_config['inference']['fire_rate'])
                update_steps = local_config['inference']['update_steps']

                # Ensure the task_id entry exists in the collector if not pre-populated
                if task_id not in submission_output_collector:
                    submission_output_collector[task_id] = [{} for _ in range(len(task_data['test_inputs']))]

                for test_idx, test_input_np_array in enumerate(task_data['test_inputs']): # these are processed inputs
                    print(f"  Predicting for test input {test_idx} of task {task_id}...")
                    # Prepare input tensor: [Batch=1, Channels, Height, Width]
                    inp_tensor = torch.tensor(test_input_np_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                    with torch.no_grad():
                        state = inp_tensor.clone()
                        for _ in range(update_steps):
                            state = model_instance(state) # Final state shape: [1, C, H, W]

                    predicted_grid = tensor_to_grid(state[0], local_config) # Pass single item tensor [C, H, W]
                    # --- Apply depadding to the predicted grid ---
                    depadded_predicted_grid = depad_grid(predicted_grid, padding_value=0) # Assuming 0 is padding

                    # Apply the transformation: subtract 1 from all integers, keep 0 as 0
                    transformed_grid = [
                        [val - 1 if val != 0 else 0 for val in row]
                        for row in depadded_predicted_grid
                    ]
                    predicted_grid = transformed_grid

                    # Update the submission_output_collector
                    # Check if the list for this task_id is long enough
                    if test_idx < len(submission_output_collector[task_id]):
                        submission_output_collector[task_id][test_idx]["attempt_1"] = predicted_grid
                        submission_output_collector[task_id][test_idx]["attempt_2"] = predicted_grid # Use same prediction for both attempts
                    else:
                        # This means test_idx is out of bounds for the pre-populated list for this task_id.
                        # This can happen if pre-population failed or if the number of test cases differs.
                        print(f"  Warning: Index {test_idx} out of range for pre-populated test cases for task {task_id} (len: {len(submission_output_collector[task_id])}). Appending new prediction entry.")
                        while len(submission_output_collector[task_id]) <= test_idx: # Ensure list is long enough
                            submission_output_collector[task_id].append({}) # Add empty dicts
                        submission_output_collector[task_id].append({"attempt_1": predicted_grid, "attempt_2": predicted_grid})

            # After processing all tasks, write the submission file
            submission_file_path = local_config['paths']['effective_output_dir'] / "submission.json"
            with open(submission_file_path, 'w') as f:
                json.dump(submission_output_collector, f) # Kaggle prefers no newlines/indentation for submission.json
            print(f"Submission file created at {submission_file_path}")

            submission_file_path = local_config['paths']['main_results_dir'] / "submission.json"
            with open(submission_file_path, 'w') as f:
                json.dump(submission_output_collector, f) # Kaggle prefers no newlines/indentation for submission.json
            print(f"Submission file created at {submission_file_path}")
            # Print the JSON content to standard output
            print("\n--- Generated submission.json content: ---")
            # print(json.dumps(submission_output_collector, indent=2)) # indent=2 for pretty printing to console
            print("--- End of submission.json content ---\n")
        else:
            print(f"Error: Unknown processing scope '{scope}' during inference.")

    else:
        print(f"Error: Invalid run_mode '{run_mode}'. Choose 'train' or 'inference'.")

    main_call_end_time = time.time()
    print(f"\n--- Main call (mode: {local_config.get('run_mode_at_execution')}) finished in: {main_call_end_time - main_call_start_time:.2f} seconds. ---")
    return local_config


# ==============================================================================
#                           SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    overall_script_start_time = time.time()

    # Helper function to serialize non-standard JSON types like Path objects
    def custom_json_serializer(obj):
        from pathlib import Path
        from matplotlib.colors import ListedColormap
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, torch.device): # Added for completeness, though not the current issue
            return str(obj)
        elif isinstance(obj, ListedColormap):
            return f"<ListedColormap: {obj.name if hasattr(obj, 'name') else 'unnamed'}>" # Or simply None or a placeholder string
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # --- First Run (could be train or inference based on global CONFIG) ---
    print("\n" + "="*80)
    print("STARTING FIRST EXECUTION PASS")
    print("="*80 + "\n")
    config_after_first_run = main()

    # --- Conditional Second Run (Automatic Inference After Training) ---
    if config_after_first_run and config_after_first_run.get('run_mode_at_execution') == 'train':
        print("\n" + "="*80)
        print("TRAINING COMPLETED. NOW STARTING AUTOMATIC INFERENCE PASS.")
        print("="*80 + "\n")

        # Prepare config for the inference run based on the completed training run's config
        inference_run_config = json.loads(json.dumps(config_after_first_run, default=custom_json_serializer)) # Deep copy with serializer
        inference_run_config['run_mode'] = 'inference'

        # If on Kaggle, switch task directories to the evaluation set for the inference pass
        if inference_run_config['environment'].get('is_kaggle', False):
            print("Switching task directories to Kaggle evaluation set for inference.")
            # User needs to define 'kaggle_evaluation_tasks_directory' in CONFIG.paths
            eval_tasks_dir = inference_run_config['paths'].get('kaggle_evaluation_tasks_directory', 'arc-agi_test_challenges') # Default if not set
            inference_run_config['batch_tasks']['tasks_directory'] = eval_tasks_dir
            inference_run_config['single_task']['arc_task_subdir'] = eval_tasks_dir # For single task inference on eval set
        # No need to change 'continue_previous_training'; inference logic handles finding the correct run.

        # Call main again for inference
        main(initial_config_override=inference_run_config)
    elif config_after_first_run:
        print(f"\nFirst execution pass completed. Mode was '{config_after_first_run.get('run_mode_at_execution', 'unknown')}'.")
        print("No automatic inference triggered (only runs after a 'train' pass).")

    overall_script_end_time = time.time()
    print(f"\nTotal script execution time (all passes): {overall_script_end_time - overall_script_start_time:.2f} seconds.")
