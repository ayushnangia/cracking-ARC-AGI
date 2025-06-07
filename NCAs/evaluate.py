# This is the evaluation script for the ARC-AGI challenge.
# Generates the result metrics and visualization.pdf files.

import json
import csv
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm

# from google.colab import drive
# drive.mount('/content/drive')

def evaluate_submission(
    submission_dict: dict,
    solutions_dict: dict,
    challenges_dict: dict,
    visualize: bool = False
) -> dict:
    """
    Compare an ARC-AGI submission against the ground truth solutions, compute per-task correctness and pixel accuracy,
    and optionally generate a PDF visualizing inputs, predictions, and ground truths.

    Parameters
    ----------
    submission_dict : dict
        Mapping from task_id to a list of dicts, each with keys "attempt_1" and "attempt_2" (predicted output grids).
    solutions_dict : dict
        Mapping from task_id to a list of ground-truth output grids or list of dicts with key "output".
    challenges_dict : dict
        Mapping from task_id to challenge data; used to extract test input grids (each challenge entry has a "test" field,
        which is a list of dicts with key "input").
    visualize : bool, optional (default=False)
        If True, generate a "visualization.pdf" in /mnt/data showing, for each test input:
        - the input grid
        - both predicted attempts
        - the ground truth grid
        Padded regions will be shown in grey for shape mismatches.

    Returns
    -------
    results : dict
        A dictionary containing:
        - mismatched_submission_ids: list of task IDs in submission but not in solutions
        - mismatched_solution_ids: list of task IDs in solutions but not in submission
        - num_total_tasks_in_both: count of tasks present in both
        - num_fully_correct_tasks: number of tasks where fraction_correct == 1.0
        - fully_correct_ids: list of task IDs that are fully correct
        - overall_pixel_accuracy: total matched pixels / total pixels (float)
        - per_task: dict mapping task_id to a dict with:
            - fraction_correct: fraction of test outputs exactly correct (0.0 to 1.0)
            - per_test_pixel_acc: list of per-test-input pixel accuracies (floats)
            - task_avg_pixel_acc: average pixel accuracy across all test inputs
    """

    # 1) Identify mismatched task IDs
    submission_ids = set(submission_dict.keys())
    solution_ids = set(solutions_dict.keys())

    mismatched_submission_ids = sorted(list(submission_ids - solution_ids))
    mismatched_solution_ids = sorted(list(solution_ids - submission_ids))
    common_task_ids = sorted(list(submission_ids & solution_ids))

    # 2) Prepare colormap for visualization (-1 => grey, 0-9 => categorical)
    cmap_colors = [
        'white',   # padding (-1)
        'black',   # 0
        'blue',    # 1
        'red',     # 2
        'green',   # 3
        'yellow',  # 4
        'grey',    # 5
        'pink',    # 6
        'orange',  # 7
        'cyan',    # 8
        'darkred'  # 9
    ]
    cmap = ListedColormap(cmap_colors)
    bounds = list(range(12))  # 0..11 for values -1+1 .. 9+1
    norm = BoundaryNorm(bounds, cmap.N)

    # Prepare PDF if visualization requested
    if visualize:
        pdf_path = "visualization.pdf"
        pdf = PdfPages(pdf_path)

    # 3) Aggregators for overall pixel counts
    total_matched_pixels = 0
    total_pixels_overall = 0

    # 4) Per-task results storage
    per_task = {}

    for task_id in common_task_ids:
        # Extract ground-truth output grids
        sol_list = solutions_dict[task_id]
        ground_truths = []
        for elem in sol_list:
            if isinstance(elem, dict) and "output" in elem:
                ground_truths.append(elem["output"])
            else:
                # Assume elem is directly a grid (list of lists)
                ground_truths.append(elem)

        # Extract submission attempts
        pred_list = submission_dict[task_id]
        # Each pred_list element is a dict with 'attempt_1' and 'attempt_2'
        # Ensure lengths match
        num_tests_truth = len(ground_truths)
        num_tests_pred = len(pred_list)
        num_tests = min(num_tests_truth, num_tests_pred)

        # Extract input grids from challenges for visualization
        challenge_entry = challenges_dict.get(task_id, {})
        test_items = challenge_entry.get("test", [])
        test_inputs = []
        for test_item in test_items:
            if isinstance(test_item, dict) and "input" in test_item:
                test_inputs.append(test_item["input"])
            else:
                # Unexpected format
                test_inputs.append(None)

        # 5) Per-test statistics
        per_test_pixel_acc = []
        num_exact_correct = 0  # count of test outputs fully matched

        for idx in range(num_tests):
            true = ground_truths[idx]
            pred_entry = pred_list[idx]
            pred1 = pred_entry.get("attempt_1", [])
            pred2 = pred_entry.get("attempt_2", [])

            # Check for exact full match
            exact1 = (pred1 == true)
            exact2 = (pred2 == true)
            if exact1 or exact2:
                num_exact_correct += 1

            # For pixel accuracy, compute matches for both attempts and take the larger
            # 6) Compute dimensions
            true_h = len(true)
            true_w = len(true[0]) if true_h > 0 else 0
            p1_h = len(pred1)
            p1_w = len(pred1[0]) if p1_h > 0 else 0
            p2_h = len(pred2)
            p2_w = len(pred2[0]) if p2_h > 0 else 0

            common_h = max(true_h, p1_h, p2_h)
            common_w = max(true_w, p1_w, p2_w)

            # 7) Padding function
            def pad_grid(grid, target_h, target_w):
                padded = [[-1] * target_w for _ in range(target_h)]
                for i in range(len(grid)):
                    for j in range(len(grid[0])):
                        padded[i][j] = grid[i][j]
                return padded

            # Pad all grids to common shape
            true_p = pad_grid(true, common_h, common_w)
            p1_p = pad_grid(pred1, common_h, common_w)
            p2_p = pad_grid(pred2, common_h, common_w)

            # Count matched pixels only over the area of the true output (true_h x true_w)
            def count_matches(padded_pred, true_grid, true_h, true_w):
                matches = 0
                for i in range(true_h):
                    for j in range(true_w):
                        if padded_pred[i][j] == true_grid[i][j]:
                            matches += 1
                return matches

            matches1 = count_matches(p1_p, true, true_h, true_w)
            matches2 = count_matches(p2_p, true, true_h, true_w)
            best_matches = max(matches1, matches2)
            total_cells = true_h * true_w

            pixel_acc = best_matches / total_cells if total_cells > 0 else 0.0
            per_test_pixel_acc.append(pixel_acc)

            # Aggregate for overall totals
            total_matched_pixels += best_matches
            total_pixels_overall += total_cells

        # 9) Summarize per-task
        fraction_correct = num_exact_correct / num_tests if num_tests > 0 else 0.0
        task_avg_pix = sum(per_test_pixel_acc) / len(per_test_pixel_acc) if per_test_pixel_acc else 0.0

        per_task[task_id] = {
            "fraction_correct": fraction_correct,
            "per_test_pixel_acc": per_test_pixel_acc,
            "task_avg_pixel_acc": task_avg_pix
        }

    # Close PDF if opened
    if visualize:
        pdf.close()

    # 10) Compute overall stats
    num_total_tasks_in_both = len(common_task_ids)
    fully_correct_ids = [
        tid for tid, info in per_task.items() if info["fraction_correct"] == 1.0
    ]
    num_fully_correct_tasks = len(fully_correct_ids)
    overall_pixel_accuracy = (
        total_matched_pixels / total_pixels_overall if total_pixels_overall > 0 else 0.0
    )

    # 11) Print summary
    print(f"**Total tasks in both files:** {num_total_tasks_in_both}")
    print(f"**Extra in submission (not in solutions):** {mismatched_submission_ids}")
    print(f"**Extra in solutions (not in submission):** {mismatched_solution_ids}")
    print(f"**Number of fully-correct tasks:** {num_fully_correct_tasks}")
    print(f"**List of fully-correct IDs:** {fully_correct_ids}")
    print(f"**Overall pixel accuracy (all tasks):** {overall_pixel_accuracy:.4f}\n")

    # 12) Build markdown table
    header = ["task_id", "fraction_correct", "task_avg_pixel_acc", "per_test_pixel_acc"]
    rows = [header]
    for tid, info in sorted(per_task.items(), key=lambda kv: kv[1]["task_avg_pixel_acc"], reverse=True):
        frac = f"{info['fraction_correct']:.4f}"
        avg_pix = f"{info['task_avg_pixel_acc']:.4f}"
        per_list = ";".join(f"{acc:.4f}" for acc in info["per_test_pixel_acc"])
        rows.append([tid, frac, avg_pix, per_list])

    # Print markdown table
    # Header
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows[1:]:
        print("| " + " | ".join(row) + " |")

    # 13) Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows[1:]:
            writer.writerow(row)

    if visualize:
        pdf_path = os.path.join(OUTPUT_DIR, "visualization.pdf")
        pdf = PdfPages(pdf_path)

        # Sort task IDs by descending task_avg_pixel_acc
        sorted_task_ids = sorted(
            per_task.keys(),
            key=lambda tid: per_task[tid]["task_avg_pixel_acc"],
            reverse=True
        )

        for task_id in sorted_task_ids:
            # (a) Extract ground‐truths & preds
            sol_list = solutions_dict[task_id]
            ground_truths = []
            for elem in sol_list:
                if isinstance(elem, dict) and "output" in elem:
                    ground_truths.append(elem["output"])
                else:
                    ground_truths.append(elem)

            pred_list = submission_dict[task_id]

            # (b) Extract test inputs
            challenge_entry = challenges_dict.get(task_id, {})
            test_items      = challenge_entry.get("test", [])
            test_inputs     = []
            for test_item in test_items:
                if isinstance(test_item, dict) and "input" in test_item:
                    test_inputs.append(test_item["input"])
                else:
                    test_inputs.append(None)

            num_tests = min(len(ground_truths), len(pred_list))

            # (c) For each test, re‐pad & plot
            for idx in range(num_tests):
                true       = ground_truths[idx]
                pred_entry = pred_list[idx]
                pred1      = pred_entry.get("attempt_1", [])
                pred2      = pred_entry.get("attempt_2", [])

                true_h = len(true)
                true_w = len(true[0]) if true_h > 0 else 0
                p1_h   = len(pred1)
                p1_w   = len(pred1[0]) if p1_h > 0 else 0
                p2_h   = len(pred2)
                p2_w   = len(pred2[0]) if p2_h > 0 else 0

                common_h = max(true_h, p1_h, p2_h)
                common_w = max(true_w, p1_w, p2_w)

                def pad_grid(grid, target_h, target_w):
                    padded = [[-1] * target_w for _ in range(target_h)]
                    for i in range(len(grid)):
                        for j in range(len(grid[0])):
                            padded[i][j] = grid[i][j]
                    return padded

                true_p = pad_grid(true, common_h, common_w)
                p1_p   = pad_grid(pred1, common_h, common_w)
                p2_p   = pad_grid(pred2, common_h, common_w)

                # Pad input
                if idx < len(test_inputs):
                    input_grid = test_inputs[idx]
                    if input_grid is not None:
                        inp_h = len(input_grid)
                        inp_w = len(input_grid[0]) if inp_h > 0 else 0
                        inp_common_h = max(inp_h, common_h)
                        inp_common_w = max(inp_w, common_w)
                        inp_p = pad_grid(input_grid, inp_common_h, inp_common_w)
                    else:
                        inp_common_h = common_h
                        inp_common_w = common_w
                        inp_p = [[-1] * inp_common_w for _ in range(inp_common_h)]
                else:
                    inp_common_h = common_h
                    inp_common_w = common_w
                    inp_p = [[-1] * inp_common_w for _ in range(inp_common_h)]

                # Plot exactly as before
                fig, axes = plt.subplots(2, 2, figsize=(6, 6))
                titles = ["Input", "Attempt 1", "Attempt 2", "Ground Truth"]
                grids  = [inp_p, p1_p, p2_p, true_p]
                for ax, title, grid_to_plot in zip(axes.flatten(), titles, grids):
                    shifted = [[cell + 1 for cell in row] for row in grid_to_plot]
                    im = ax.imshow(shifted, cmap=cmap, norm=norm)
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("black")
                        spine.set_linewidth(0.5)

                plt.suptitle(f"Task {task_id} | Test #{idx+1}", fontsize=10)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

        # Finally, close the PDF
        pdf.close()

    # 14) Return dictionary of results
    return {
        "mismatched_submission_ids": mismatched_submission_ids,
        "mismatched_solution_ids": mismatched_solution_ids,
        "num_total_tasks_in_both": num_total_tasks_in_both,
        "num_fully_correct_tasks": num_fully_correct_tasks,
        "fully_correct_ids": fully_correct_ids,
        "overall_pixel_accuracy": overall_pixel_accuracy,
        "per_task": per_task
    }


# --- Example Usage ---

# 1. Define the same paths used by your NCA solver
ARC_DATA_DIR = "../dataset/script-tests/grouped-tasks"  # Or your custom path
OUTPUT_DIR = "./runs/prl_test_250607_133838"                 # Or your custom path

# The submission file generated by your NCA script
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.json")

# The original challenge file that your NCA script read as input
# IMPORTANT: This must be the SAME file your solver used.
CHALLENGES_FILE = os.path.join(ARC_DATA_DIR, "challenges.json")

# The solutions file corresponding to the challenges file
SOLUTIONS_FILE = os.path.join(ARC_DATA_DIR, "solutions.json")


# 2. Load the files from the correct paths
print(f"Loading submission from: {SUBMISSION_FILE}")
print(f"Loading challenges from: {CHALLENGES_FILE}")
print(f"Loading solutions from: {SOLUTIONS_FILE}")

try:
    with open(SUBMISSION_FILE) as f:
        submission_dict = json.load(f)
    with open(SOLUTIONS_FILE) as f:
        solutions_dict = json.load(f)
    with open(CHALLENGES_FILE) as f:
        challenges_dict = json.load(f)
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Could not find a required file.")
    print(f"Missing file: {e.filename}")
    print("Please ensure the paths at the top of this script are correct.")
    exit()

# 3. Call the evaluator function (no changes here)
results = evaluate_submission(
    submission_dict=submission_dict,
    solutions_dict=solutions_dict,
    challenges_dict=challenges_dict,
    visualize=True  # Set to True to get the helpful PDF
)

# 4. Update the final print statements to show correct output paths
print(f"\nEvaluation complete.")
print(f"CSV report saved to: results.csv")
print(f"Visualization PDF saved to: visualization.pdf")