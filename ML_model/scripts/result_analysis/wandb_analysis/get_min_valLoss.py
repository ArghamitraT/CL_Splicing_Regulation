import wandb
import pandas as pd

# --- CONFIGURATION ---
# 1. Set your project path: "entity_name/project_name"
#    Based on your image, this is likely:
PROJECT_PATH = "at3836-columbia-university/Psi__SWEEP"

# 2. Set the exact metric name for validation loss
#    You MUST check wandb for the exact name.
#    Common names: "val_loss", "validation_loss", "val/loss"
METRIC_NAME = "val_loss_epoch"

# 3. Set the name of your epoch column
#    This is usually "epoch" or "_step"
EPOCH_COLUMN = "epoch"
# ---------------------


def find_best_run(project_path, metric_name, epoch_col):
    """
    Finds the run with the minimum value for a given metric
    in a Weights & Biases project.
    """
    print(f"Connecting to Weights & Biases...")
    try:
        api = wandb.Api()
        # Get all runs from the specified project
        runs = api.runs(path=project_path)
    except Exception as e:
        print(f"Error connecting to W&B. Have you run 'wandb login'?")
        print(f"Details: {e}")
        return

    best_run_name = None
    best_loss = float('inf')
    best_epoch = -1
    
    print(f"Found {len(runs)} runs in '{project_path}'.")
    print(f"Searching for minimum '{metric_name}'...")

    for run in runs:
        try:
            # Get the history for the run, only fetching the metrics we need
            history = run.history(keys=[metric_name, epoch_col])

            # Check if the metric exists in this run
            if metric_name not in history.columns:
                # print(f"Warning: Metric '{metric_name}' not in run {run.name}. Skipping.")
                continue
                
            # Drop any NaN values from the metric column to avoid errors
            history = history.dropna(subset=[metric_name])
            if history.empty:
                continue

            # Find the minimum value for the metric in this run
            min_loss_for_run = history[metric_name].min()

            # Check if this run's min is the new overall min
            if min_loss_for_run < best_loss:
                best_loss = min_loss_for_run
                best_run_name = run.name
                
                # Get the epoch at which this minimum loss occurred
                best_epoch_index = history[metric_name].idxmin()
                best_epoch = history.loc[best_epoch_index][epoch_col]
                
                print(f"  New best found! Run: {best_run_name}, Loss: {best_loss:.6f}")

        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue

    # After checking all runs, print the final result
    print("\n--- Search Complete ---")
    if best_run_name:
        print(f"🏆 Best Run Found: {best_run_name}")
        print(f"Lowest Validation Loss: {best_loss:.6f}")
        print(f"Occurred at Epoch: {int(best_epoch)}")
    else:
        print(f"Could not find any runs with the metric '{metric_name}'.")
        print(f"Please double-check your METRIC_NAME and EPOCH_COLUMN settings.")

if __name__ == "__main__":
    find_best_run(PROJECT_PATH, METRIC_NAME, EPOCH_COLUMN)