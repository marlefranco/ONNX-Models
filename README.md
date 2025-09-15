# ONNX-Models

## Introduction
This repository consolidates the spectral classification pipelines that power the TissueSense and stone-identification workflows across Python, ONNX, and MATLAB environments.【F:TS_ModelPrediction/main.py†L16-L87】【F:Python/TS_MatlabMigration/main.py†L34-L191】【F:MATLAB/tissue_sense/ml-framework/README.md†L1-L43】 It contains code for preprocessing hyperspectral measurements, extracting engineered spectral features, training traditional and deep learning models, and serving optimized ONNX classifiers for deployment.【F:TS_ModelPrediction/processing_module.py†L10-L355】【F:TS_ModelPrediction/ml_framework/powerRatioFeatures.py†L163-L232】【F:Python/TS_MatlabMigration/ml_framework/classification.py†L11-L121】【F:Python/TS_MatlabMigration/ml_utility/cnn.py†L13-L138】

## Repository Structure
- `TS_ModelPrediction/` – Production inference pipeline that reads raw spectrometer exports, applies reference subtraction and filtering, builds power-ratio feature sets, and evaluates ONNX classifiers selected by light source and automatic brightness status.【F:TS_ModelPrediction/main.py†L16-L87】【F:TS_ModelPrediction/processing_module.py†L10-L347】 The directory also holds configurable ratios in `config.yaml` and pre-trained ONNX models resolved as `<SOURCE>_<AB_STATUS>.onnx`.【F:TS_ModelPrediction/config.yaml†L3-L27】【F:TS_ModelPrediction/main.py†L79-L87】
- `Python/TS_MatlabMigration/` – Research environment for migrating MATLAB feature engineering into Python, including data import helpers, ANOVA- and peak-based feature discovery, classical ML model selection with cross-validation, and CNN-based saliency exploration.【F:Python/TS_MatlabMigration/main.py†L3-L191】【F:Python/TS_MatlabMigration/ml_framework/classification.py†L11-L121】【F:Python/TS_MatlabMigration/ml_utility/cnn.py†L13-L138】
- `Python/Noise_Study/` – Standalone notebook-style script for benchmarking filtering strategies by computing SNR, SPNR, Pearson correlation, and phase-shift metrics on dark-reference corrected spectra.【F:Python/Noise_Study/Nose_Study.py†L1-L140】【F:Python/Noise_Study/Nose_Study.py†L189-L199】
- `MATLAB/stone_id` and `MATLAB/tissue_sense/` – Original MATLAB projects that use the MathWorks ML Framework to configure autoML experiments for supervised spectral classification, complete with project templates and pipeline tooling.【F:MATLAB/stone_id/ml-framework/README.md†L1-L50】【F:MATLAB/tissue_sense/ml-framework/README.md†L1-L48】

## Python Workflows

### TS_ModelPrediction ONNX pipeline
1. **Configure input paths and power ratios** – Update `config.yaml` with the acquisition directories, integration time, light source, emission mode, AB status, and custom numerator/denominator wavelength windows (specified as either four-number lists or `"start-end"` pairs).【F:TS_ModelPrediction/config.yaml†L3-L27】【F:TS_ModelPrediction/main.py†L45-L58】
2. **Preprocess spectra** – `process_directory` parses each `.txt` export, filters by integration time, subtracts either cached dark references or row means, applies FIR smoothing, interpolates to a standard 400–940 nm grid, normalizes to the 635–641 nm window, and filters out low-signal rows before returning stacked spectra and labels.【F:TS_ModelPrediction/processing_module.py†L10-L347】 Supported subtraction modes are `darkref` and `avg` so the same pipeline works with legacy and averaged reference files.【F:TS_ModelPrediction/processing_module.py†L240-L333】
3. **Feature engineering** – `calculate_spectral_features` converts spectra into configurable power ratios and augments them with AUC, peak-to-trough, standard deviation, and mean intensity statistics per wavelength band, producing the feature table expected by both ONNX and scikit-learn models.【F:TS_ModelPrediction/ml_framework/powerRatioFeatures.py†L163-L232】
4. **Run inference** – The script resolves `<SOURCE>_<AB_STATUS>.onnx`, executes ONNX Runtime, and reports accuracy, error, sensitivity, specificity, precision, F1, and AUC when available.【F:TS_ModelPrediction/main.py†L79-L87】【F:TS_ModelPrediction/processing_module.py†L358-L428】 Results use the same label encoder as the training workflow so confusion matrices align with Tissue vs. Stone conventions.【F:TS_ModelPrediction/main.py†L85-L87】【F:TS_ModelPrediction/processing_module.py†L381-L428】

### Python training and research utilities
- `Python/TS_MatlabMigration/main.py` orchestrates importing TRL5 spectral datasets, balances labels, partitions spectra by filename to avoid leakage, and prepares both engineered features and raw spectra for downstream analysis.【F:Python/TS_MatlabMigration/main.py†L34-L191】 The script demonstrates running wavelet transforms, peak detection, saliency mapping, and plotting routines ported from MATLAB.
- `ml_framework/classification.py` evaluates KNN, decision tree, random forest, and XGBoost models through `GridSearchCV`, tracks overfitting penalties, and prints sensitivity/specificity plus error summaries for the best cross-validated classifier.【F:Python/TS_MatlabMigration/ml_framework/classification.py†L11-L121】
- `ml_utility/cnn.py` defines a 1D CNN that ingests paired intensity/wavelength channels, trains with binary cross entropy, and exposes saliency and Grad-CAM utilities to highlight informative wavelength regions.【F:Python/TS_MatlabMigration/ml_utility/cnn.py†L13-L138】 These helpers back the exploratory calls in the migration script for visualizing tissue vs. non-tissue cues.【F:Python/TS_MatlabMigration/main.py†L123-L144】

### Noise characterization
`Python/Noise_Study/Nose_Study.py` provides a reproducible workflow for subtracting dark references, trimming wavelength ranges, and quantifying filtering performance with MSE, signal distortion, SNR, SPNR, correlation, and phase shift metrics – helpful when tuning preprocessing stages before model training.【F:Python/Noise_Study/Nose_Study.py†L13-L140】【F:Python/Noise_Study/Nose_Study.py†L189-L199】

## MATLAB Projects
Both MATLAB subprojects embed the MathWorks ML Framework so you can scaffold experiments, sweep preprocessing parameters, and persist optimal learners directly from MATLAB. Use the provided project templates (`createProjectFromTemplate`) and pipeline generators (`newPipelineTemplate`) to recreate the autoML experiments that produced the reference ONNX models.【F:MATLAB/stone_id/ml-framework/README.md†L1-L50】【F:MATLAB/tissue_sense/ml-framework/README.md†L1-L48】

## Getting Started

### Environment setup
1. Install Python 3.9+ and create a virtual environment.
2. Install dependencies used across the pipelines: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `pyyaml`, `onnxruntime`, `onnxmltools`, `imbalanced-learn`, `xgboost`, `tensorflow`, and `torch` (plus any GPU variants required for your hardware). These packages satisfy the imports in the ONNX, migration, CNN, and noise scripts.【F:TS_ModelPrediction/main.py†L2-L12】【F:TS_ModelPrediction/processing_module.py†L1-L7】【F:TS_ModelPrediction/ml_framework/powerRatioFeatures.py†L163-L232】【F:Python/TS_MatlabMigration/main.py†L3-L26】【F:Python/TS_MatlabMigration/ml_framework/classification.py†L1-L48】【F:Python/TS_MatlabMigration/ml_utility/cnn.py†L2-L138】【F:Python/Noise_Study/Nose_Study.py†L1-L7】
3. For MATLAB workflows, open the corresponding project in MATLAB R2020a or later and restore path dependencies using the included project definition files.【F:MATLAB/tissue_sense/ml-framework/README.md†L1-L48】

### Running ONNX inference
1. Export the spectrometer `.txt` files and corresponding averaged dark references into directories referenced by `config.yaml`.
2. Adjust integration time, light source, emission mode, AB status, and ratio definitions in the config to match your acquisition campaign.【F:TS_ModelPrediction/config.yaml†L3-L27】
3. Execute `python TS_ModelPrediction/main.py` (optionally passing an alternative config path). The script prints label counts, filtering summaries, and ONNX evaluation metrics to the console.【F:TS_ModelPrediction/main.py†L16-L87】【F:TS_ModelPrediction/processing_module.py†L240-L428】

### Training and experimentation in Python
1. Place training data under `Python/TS_MatlabMigration/data/` following the TRL5 folder conventions referenced in `main.py`.
2. Run `python Python/TS_MatlabMigration/main.py` to reproduce the preprocessing, partitioning, feature extraction, CNN saliency, and model selection workflow.【F:Python/TS_MatlabMigration/main.py†L34-L191】
3. Inspect the printed cross-validation results, evaluation metrics, and plots to decide which engineered features or ratios to export for ONNX conversion.【F:Python/TS_MatlabMigration/ml_framework/classification.py†L62-L121】【F:Python/TS_MatlabMigration/main.py†L140-L190】

## Build and Test
There are no automated tests in this repository. Validate changes by running the relevant entry points (`python TS_ModelPrediction/main.py` or `python Python/TS_MatlabMigration/main.py`) against representative datasets and verifying that preprocessing, feature extraction, and evaluation complete without errors.【F:TS_ModelPrediction/main.py†L16-L87】【F:Python/TS_MatlabMigration/main.py†L34-L191】

## Contribute
1. Fork the repository and create feature branches per enhancement.
2. Keep Python formatting consistent and favor parameterized configs so ONNX and MATLAB assets stay in sync.【F:TS_ModelPrediction/main.py†L16-L87】【F:Python/TS_MatlabMigration/main.py†L34-L191】
3. Provide documentation updates for new pipelines or datasets so downstream teams can reproduce spectral preprocessing and model evaluation steps.

## Additional Resources
- MathWorks ML Framework documentation is included in each MATLAB project (`ml-framework/README.md`).【F:MATLAB/stone_id/ml-framework/README.md†L1-L50】【F:MATLAB/tissue_sense/ml-framework/README.md†L1-L48】
- Pre-trained ONNX models (`TS_ModelPrediction/ONNX Models/`) are named by `<SOURCE>_<AB_STATUS>.onnx`; ensure new exports follow the same convention for compatibility with `main.py`.【F:TS_ModelPrediction/main.py†L79-L87】
