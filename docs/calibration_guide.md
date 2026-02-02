# Prediction Calibration System Guide

## Overview
The Calibration System addresses systematic bias in model predictions (e.g., persistent under-prediction of points) by implementing a multi-layered correction pipeline. It consists of:
1. **Bias Tracking**: Storing predictions vs actual outcomes.
2. **Analysis**: Detecting systematic over/under variations.
3. **Training Correction**: Penalizing bias directly in the loss function.
4. **Inference Correction**: Applying scalar multipliers to final predictions to align distributions.

## Components

### 1. Calibration Tracker (`src/calibration_tracker.py`)
Core module that manages the historical data and factor calculations.
- **History File**: `data/calibration/prediction_history.csv`
- **Factors File**: `data/calibration/calibration_factors.json`

### 2. Training Integration
The Gaussian Negative Log Likelihood loss function in `train_models.py` now accepts a `calibration_penalty_weight`.
- **`--calibrate`**: Flag to enable this penalty during training.
- **Penalty Logic**: Penalizes the squared mean residual ($(\bar{y} - \bar{\hat{y}})^2$) across the batch, forcing the model to center its distribution on the true mean.

### 3. Inference Adjustment
`batch_predict.py` loads calibration factors and applies them to the raw model output *before* making betting decisions.
- **Global Factors**: Per-stat multipliers (e.g., `PTS * 1.05`).
- **Player Factors**: High-volume players get specific corrections if they deviate from the global trend.

## Workflow

### Daily Routine
1. **Backfill Actuals**: Run `scripts/update_actuals.py` to fill in yesterday's results.
2. **Recalibrate**: Run `scripts/update_calibration.py` to compute new factors based on recent trends (last 90 days).
   - If `Under Rate` > 65% for a stat, a multiplier > 1.0 is generated.
3. **Predict**: `batch_predict.py` automatically uses the latest factors.

### Periodic Re-Training
Once a week, run `scripts/calibrate_models.py` to fine-tune the model weights themselves using the calibration penalty. This "bakes in" the correction so the scalar multipliers don't have to work as hard.

## Interpretation of Report

Check `data/calibration/calibration_report.md`:
- **Under Bias %**: Percentage of time the actual value exceeded the prediction. Target is 50%.
    - `> 60%`: Systematic Under-prediction (Model is too conservative).
    - `< 40%`: Systematic Over-prediction (Model is too aggressive).
- **Factor**: The multiplier being applied. `1.10` means predictions are boosted by 10%.

## Troubleshooting
- **Zero Factors**: If factors are 1.0, ensure `update_calibration.py` has run and `prediction_history.csv` has valid `actual` values.
- **Over-correction**: If the model starts over-predicting, check if the `calibration_penalty_weight` is too high (default 0.1) or if the manual multiplier cap (1.25) needs adjustment in `calibration_tracker.py`.
