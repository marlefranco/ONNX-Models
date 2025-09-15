# main.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#from ml_framework.classification import evaluate_model_on_test_set, train_models_with_cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Switches to a more stable backend for VS Code
from ml_framework.powerRatioFeatures import calculate_spectral_features, plot_power_ratio_histograms
from onnxmltools.convert.common.data_types import FloatTensorType
import yaml
from processing_module import process_directory, evaluate_onnx_model
import os
import sys


def main(config_path=None):
    # Resolve config path robustly:
    # 1) If provided, use it; 2) try alongside this script; 3) fall back to CWD.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if config_path is None:
        candidates = [
            os.path.join(script_dir, "config.yaml"),
            os.path.join(os.getcwd(), "config.yaml"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                config_path = p
                break
    elif not os.path.isabs(config_path):
        # Allow relative paths from CWD or script dir
        explicit_candidates = [
            config_path,
            os.path.join(script_dir, config_path),
            os.path.join(os.getcwd(), config_path),
        ]
        for p in explicit_candidates:
            if os.path.isfile(p):
                config_path = os.path.abspath(p)
                break

    if config_path is None or not os.path.isfile(config_path):
        looked = candidates if candidates else [config_path or "config.yaml"]
        raise FileNotFoundError(
            f"Config file not found. Looked for: {', '.join(map(str, looked))}. "
            f"Pass a path: python {os.path.basename(__file__)} TS_ModelPrediction/config.yaml"
        )

    # === Load config ===
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    main_folder = config.get("main_folder")
    darkref_folder = config.get("darkref_folder")
    Reference_Sub = config.get("Sub")
    integration_time = str(config.get("integration_time", ""))
    source = str(config.get("source", "")).upper()
    emission = str(config.get("emission", "")).upper()
    power_ratios = config.get("power_ratios", {})
    ab_status = config.get("ab_status", "AB_OFF").upper()  # New parameter for AB status

    print(f"\n[Processing] Source: {source} | Main: {main_folder} | Darkref: {darkref_folder}")
    X_Test, Y_Test, wavelength_df = process_directory(
        main_folder=main_folder,
        darkref_folder=darkref_folder,
        integration_time=integration_time,
        source=source,
        Reference_Sub=Reference_Sub,
        emission=emission
    )
    Y_Test = pd.Series(Y_Test)
    print("Label counts:\n", Y_Test.value_counts())


    # Transform user-provided power_ratios from config into the required format for feature extraction
    # Expecting: power_ratios = {"Ratio 1": [465, 485, 515, 535], ...} or similar
    ratio_ranges = {}
    for key, val in power_ratios.items():
        if isinstance(val, (list, tuple)) and len(val) == 4:
            # Convert [start1, end1, start2, end2] to ("start1-end1", "start2-end2")
            range1 = f"{val[0]}-{val[1]}"
            range2 = f"{val[2]}-{val[3]}"
            ratio_ranges[key] = (range1, range2)
        elif isinstance(val, dict) and "range1" in val and "range2" in val:
            ratio_ranges[key] = (val["range1"], val["range2"])
        else:
            # Already in correct format or unknown, just pass through
            ratio_ranges[key] = val
        

    # Compute power ratio features
    power_ratio_features_test= calculate_spectral_features(X_Test, Y_Test, wavelength_df, ratio_ranges)


        # Strip and rename to standardized names
    power_ratio_features_test.columns = power_ratio_features_test.columns.str.strip()

    # Optional: Rename for consistency
    power_ratio_features_test = power_ratio_features_test.rename(columns={

        "ratio_1": "Ratio 1",
        "ratio_2": "Ratio 2"
    })

    X_test_knn = power_ratio_features_test.iloc[:, :-1]
    X_test_knn.fillna(X_test_knn.mean(), inplace=True)# First two columns (features)
    y_test_knn = power_ratio_features_test.iloc[:, -1]   # Third column (labels)
    
    # Build ONNX model filename based on source and ab_status
    onnx_model_name = f"{source}_{ab_status}.onnx"
    onnx_model_path = os.path.join(script_dir, "ONNX Models", onnx_model_name)
    if not os.path.exists(onnx_model_path):
        print(f"[Warning] ONNX model not found: {onnx_model_path}. Skipping ONNX evaluation.")
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(Y_Test)
        evaluate_onnx_model(onnx_model_path, X_test_knn, y_test_knn, label_encoder)
    
    
if __name__ == "__main__":
    # Optional CLI: python TS_ModelPrediction/main.py [config_path]
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
