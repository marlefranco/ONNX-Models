# TS Model Prediction Pipeline

This guide walks through preparing a Python environment, configuring the TissueSense ONNX inference pipeline, and executing it against your spectral datasets.

## 1. Create and activate a Python 3.9+ virtual environment
1. Install Python 3.9 or newer.
2. From the repository root (`ONNX-Models/`), create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
3. Activate the environment before installing any dependencies:
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

## 2. Install pipeline dependencies
With the virtual environment active, install the packages required by the ONNX inference workflow:
```bash
pip install --upgrade pip
pip install -r TS_ModelPrediction/requirements.txt
```
Run these commands from the `ONNX-Models/` directory so the relative path to `requirements.txt` resolves correctly.

## 3. Configure `config.yaml`
`TS_ModelPrediction/config.yaml` holds the runtime settings the pipeline consumes. Update the following fields to reflect your acquisition setup:

- **`main_folder`** – Absolute path to the directory that contains the spectrometer `.txt` exports you want to evaluate.
- **`darkref_folder`** – Absolute path to the folder holding the matching dark-reference captures (or leave blank if `Sub` is set to `avg`).
- **`integration_time`** – Integration time filter (e.g., `1000`, `2000`, `3000`, or `ALL`).
- **`source`** – Light source identifier (`LED` or `XENON`). This value also selects the ONNX model file.
- **`emission`** – Measurement mode (`Emission`, `NonEmission`, or `ALL`).
- **`ab_status`** – Automatic brightness flag (`AB_ON` or `AB_OFF`). Combined with `source`, it determines the ONNX model filename (`<SOURCE>_<AB_STATUS>.onnx`).
- **`Sub`** – Reference subtraction strategy (`darkref` to subtract captured dark references, `avg` to subtract the spectrum mean).
- **`power_ratios`** – Dictionary defining numerator and denominator wavelength windows. Provide each ratio as four numbers (`[start1, end1, start2, end2]`) or explicit ranges (`{"range1": "465-485", "range2": "515-535"}`) to align with your model training assumptions.

## 4. Understand the ONNX model layout
Pre-trained classifiers live in `TS_ModelPrediction/ONNX Models/`. Each file follows the `<SOURCE>_<AB_STATUS>.onnx` naming convention (for example, `LED_AB_ON.onnx`). Ensure your configuration values match the file you expect the pipeline to load.

## 5. Run the inference pipeline
From the `ONNX-Models/` directory—and with your virtual environment activated—execute:
```bash
python TS_ModelPrediction/main.py --config TS_ModelPrediction/config.yaml
```
Adjust the `--config` argument if you keep alternative configuration files. The script ingests the configured directories, computes power-ratio features, selects the appropriate ONNX model, and prints evaluation metrics to the console.
