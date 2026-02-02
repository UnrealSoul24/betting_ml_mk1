
import os
import sys
import subprocess
import argparse

def run_step(description, command):
    print(f"\n=== {description} ===")
    ret = subprocess.call(command, shell=True)
    if ret != 0:
        print(f"Error executing: {command}")
        sys.exit(ret)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', action='store_true', help='Run full automated loop')
    args = parser.parse_args()
    
    # Define paths
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    script_actuals = os.path.join(root, 'scripts', 'update_actuals.py')
    script_calib = os.path.join(root, 'scripts', 'update_calibration.py')
    script_train = os.path.join(root, 'src', 'train_models.py')
    
    # 1. Update Actuals
    # run_step("Backfilling Actuals", f"python {script_actuals}")
    # (Commented out default run to verify connectivity first, or user can run separate. 
    # But wrapper implies doing it. I'll uncomment.)
    run_step("Backfilling Actuals", f"python {script_actuals}")
    
    # 2. Update Calibration Factors
    run_step("Updating Calibration Factors", f"python {script_calib}")
    
    # 3. Retrain (Fine-tune) Global Model with Calibration Penalty
    # We fine-tune the global model
    # Note: --calibrate flag enables the penalty
    run_step("Fine-Tuning Global Model with Calibration", f"python {script_train} --pretrain --calibrate --calibration_weight 0.1")

    print("\n=== Calibration Cycle Complete ===")

if __name__ == "__main__":
    main()
