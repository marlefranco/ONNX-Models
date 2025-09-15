import os
import numpy as np
import pandas as pd
from scipy.signal import firwin, lfilter
from scipy.interpolate import interp1d
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# === 1. Read main file ===
def read_main_file(file_path, integration_time=None):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  

    data_groups = []
    meta_groups = []
    pixel_data_idx = None
    header = None
    for line in lines:
        items = [val.strip() for val in line.split(',')]
        if 'PixelDataArray' in items:
            pixel_data_idx = items.index('PixelDataArray')
            header = items
            # Extract wavelengths from the header row (all columns after PixelDataArray)
            wavelengths = [float(val) for val in items[pixel_data_idx+1:] if val != '']
            continue  # Do not skip this header row for index finding
        if pixel_data_idx is None:
            continue
        # Only process rows with enough columns
        if len(items) <= pixel_data_idx:
            continue
        try:
            float_values = [float(val) for val in items[pixel_data_idx+1:] if val != '']
        except ValueError:
            float_values = []
        if float_values:
            data_groups.append(float_values)
            meta_groups.append(items[:pixel_data_idx+1])

    if not data_groups or not header or not wavelengths:
        raise ValueError(f"No valid spectra or header found in {file_path}")

    # Find IntegrationTime index in the header
    if 'IntegrationTime' in header:
        int_time_idx = header.index('IntegrationTime')
    else:
        int_time_idx = 30  # fallback

    # Filter by integration_time if specified
    if integration_time is not None:
        filtered_intensities = []
        filtered_metadata = []
        for i, meta in enumerate(meta_groups):
            if len(meta) > int_time_idx:
                try:
                    meta_time = float(meta[int_time_idx])
                    target_time = float(integration_time)
                    if meta_time == target_time:
                        filtered_intensities.append(data_groups[i])
                        filtered_metadata.append(meta)
                except Exception as e:
                    continue
        intensities = filtered_intensities
        metadata = filtered_metadata
    else:
        intensities = data_groups
        metadata = meta_groups
    return wavelengths, intensities, metadata


# === Load averaged dark reference file (handles both styles) ===
def load_averaged_darkref(folder_path, main_wavelengths=None, integration_time=None):
    """
    Reads a dark reference file:
      - Skips FILE_START/FILE_END
      - Handles both formats:
          (a) groups separated by blank lines
          (b) continuous rows without blank lines
      - Splits wavelength and intensities
      - Averages intensities for specified integration time if provided
    Returns: wavelengths, avg_intensity
    """
    import os
    import numpy as np

    # Find the first .txt file
    darkref_file = None
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.txt'):
            darkref_file = os.path.join(folder_path, fname)
            break
    if darkref_file is None:
        raise FileNotFoundError(f"No .txt dark reference file found in {folder_path}")

    # Read all non-empty lines
    with open(darkref_file, 'r') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    # Remove FILE_START/FILE_END if present
    if raw_lines and raw_lines[0].upper().startswith('FILE_START'):
        raw_lines = raw_lines[1:]
    if raw_lines and raw_lines[-1].upper().startswith('FILE_END'):
        raw_lines = raw_lines[:-1]

    # === Step 1: Try to group by blank lines (multi-block format) ===
    groups = []
    current_group = []
    with open(darkref_file, 'r') as f:
        for line in f:
            if not line.strip():  # blank line
                if current_group:
                    groups.append(current_group)
                    current_group = []
            else:
                current_group.append(line.strip())
    if current_group:
        groups.append(current_group)

    # If we only got one big group → it means file has no blank separators
    if len(groups) == 1:
        # Treat first row as wavelengths, rest as intensity rows
        all_values = []
        for line in groups[0]:
            items = [val.strip() for val in line.split(',') if val.strip()]
            all_values.append(items)

        # Remove metadata (first 16 cols) from each row
        data_groups = []
        for row in all_values:
            try:
                float_values = [float(v) for v in row[16:]]
                data_groups.append(float_values)
            except ValueError:
                continue
    else:
        # Multi-group format (first group = wavelengths, rest = intensities)
        data_groups = []
        for group in groups:
            values = []
            for line in group:
                items = [val.strip() for val in line.split(',') if val.strip()]
                values.extend(items)
            # Strip metadata
            values = values[16:]
            float_values = []
            for v in values:
                try:
                    float_values.append(float(v))
                except ValueError:
                    continue
            if float_values:
                data_groups.append(float_values)

    if not data_groups:
        raise ValueError(f"No valid dark reference data found in {darkref_file}")

    wavelengths = np.array(data_groups[0], dtype=float)
    intensities = np.array(data_groups[1:], dtype=float)

    # Validate wavelength match
    if main_wavelengths is not None:
        if not np.isclose(wavelengths[0], main_wavelengths[0], atol=1e-2):
            print(f"[Warning] Wavelength start mismatch: DarkRef={wavelengths[0]}, Main={main_wavelengths[0]}")
        if len(wavelengths) != len(main_wavelengths):
            print(f"[Warning] Wavelength count mismatch: DarkRef={len(wavelengths)}, Main={len(main_wavelengths)}")

    # === Handle averaging by integration time if requested ===
    avg_intensity = None
    if integration_time is not None:
        matching_rows = []
        for group in groups[1:]:
            meta = [val.strip() for val in group[0].split(',')[:16]]
            try:
                row_int_time = float(meta[6])  # assume integration time at column 6
            except Exception:
                continue
            if str(row_int_time) == str(integration_time):
                values = []
                for line in group:
                    items = [val.strip() for val in line.split(',') if val.strip()]
                    values.extend(items[16:])
                float_values = []
                for v in values:
                    try:
                        float_values.append(float(v))
                    except ValueError:
                        continue
                if float_values:
                    matching_rows.append(float_values)
        if matching_rows:
            avg_intensity = np.mean(np.array(matching_rows, dtype=float), axis=0)
        else:
            print(f"[Warning] No dark reference rows found for IntegrationTime={integration_time}")
            avg_intensity = np.mean(intensities, axis=0)
    else:
        avg_intensity = np.mean(intensities, axis=0)

    return wavelengths, avg_intensity



# === Subtract background ===
def subtract_background(main_groups, background_avg):
    return np.array(main_groups) - np.array(background_avg)


# === 4. Filtering & preprocessing ===
def apply_fir_filter(data, sample_rate=None, cutoff_freq=10, numtaps=101):
    nyquist = sample_rate / 2
    fir_coeff = firwin(numtaps, cutoff_freq / nyquist)
    return lfilter(fir_coeff, 1.0, data)

def interpolate_to_standard(wavelength, intensity, new_range):
    f = interp1d(wavelength, intensity, kind='linear', fill_value="extrapolate")
    return f(new_range)

def normalize_spectra(interpolated_intensity, wavelength):
    # Ensure 2D (n_samples, n_wavelengths)
    if interpolated_intensity.ndim == 1:
        interpolated_intensity = interpolated_intensity[np.newaxis, :]

    # Offset correction
    largest_negative = np.min(interpolated_intensity, axis=1)
    offset_corrected = interpolated_intensity + (0.1 - largest_negative[:, np.newaxis])

    # Mask for 635–641 nm
    mask = (wavelength >= 635) & (wavelength <= 641)
    if not np.any(mask):
        raise ValueError("No wavelength values found in the 635–641 nm range.")

    # Median intensity across 635–641 nm for each spectrum
    normalization_factor = np.median(offset_corrected[:, mask], axis=1)

    # Normalize
    normalized_spectra = offset_corrected / normalization_factor[:, np.newaxis]

    return normalized_spectra


# === 5. Main processing + merging ===
def process_directory(main_folder, darkref_folder, integration_time, source, emission, Reference_Sub="darkref"):
    """
    Reference_Sub:
        - "darkref": subtract dark reference file (cached by integration time)
        - "avg": subtract row-wise average intensity
        - "none": no subtraction
    """
    folder_path = main_folder
    source_only = source
    all_spectra = []
    all_labels = []

    kept_count = 0
    skipped_count = 0

    # === Cache for darkref averages keyed by integration time ===
    darkref_cache = {}

    def filter_spectra(wavelengths, spectrum, source_type, emission_type):
        mask = (wavelengths >= 750) & (wavelengths <= 900)
        row_max = spectrum[mask].max()
        if source_type == 'LED':
            threshold = 0.3 if emission_type == 'NONEMISSION' else 0.1
        elif source_type == 'XENON':
            threshold = 0.1 if emission_type == 'NONEMISSION' else 0.08
        else:
            threshold = 0.1
        return row_max < threshold

    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            continue

        main_file = os.path.join(folder_path, file)
        main_wavelengths, main_intensities, metadata = read_main_file(main_file, integration_time=integration_time)

        # Check for empty metadata or insufficient columns
        if not metadata or not metadata[0]:
            print(f"[Warning] Skipping {file}: no valid metadata rows found.")
            continue

        # Dynamically find indices for required metadata fields
        meta_row = metadata[0]
        ab_status_idx = meta_row.index('dropdownAB') if 'dropdownAB' in meta_row else 6
        source_idx = meta_row.index('lightSourceType') if 'lightSourceType' in meta_row else 12
        label_idx = meta_row.index('targetType') if 'targetType' in meta_row else 18
        int_time_idx = meta_row.index('IntegrationTime') if 'IntegrationTime' in meta_row else 30

        # Check index bounds
        if any(idx >= len(meta_row) for idx in [ab_status_idx, source_idx, label_idx, int_time_idx]):
            print(f"[Warning] Skipping {file}: metadata row does not have enough columns.")
            continue

        ab_status = meta_row[ab_status_idx]
        file_source = meta_row[source_idx].split()[-1].upper().strip("()")
        new_wavelength_range = np.linspace(400, 940, len(main_wavelengths))

        # Skip if source mismatch
        if file_source != source_only:
            print(f"Skipping {file} due to source mismatch: {file_source} vs {source_only}")
            continue

        for idx, intensity in enumerate(main_intensities):
            label = metadata[idx][label_idx].upper()
            int_time = metadata[idx][int_time_idx]

            if label == "UNKNOWN":
                continue

            final_label = "Stone" if label in ["COM", "UA", "BEGO"] else "Tissue"

            # === Reference subtraction ===
            if Reference_Sub.lower() == "darkref":
                # Use cached darkref if available
                if int_time not in darkref_cache:
                    background_wl, background_avg = load_averaged_darkref(
                        darkref_folder,
                        main_wavelengths=main_wavelengths,
                        integration_time=int_time
                    )
                    darkref_cache[int_time] = (background_wl, background_avg)
                else:
                    background_wl, background_avg = darkref_cache[int_time]

                sub_intensity = subtract_background(intensity, background_avg)

            elif Reference_Sub.lower() == "avg":
                row_mean = np.mean(intensity)
                sub_intensity = np.array(intensity) - row_mean

            else:
                raise ValueError(f"Invalid Reference_Sub: {Reference_Sub}")

            # === Preprocessing ===
            filtered = apply_fir_filter(sub_intensity, sample_rate=len(sub_intensity))
            interpolated = interpolate_to_standard(np.array(main_wavelengths), filtered, new_wavelength_range)
            normalized = normalize_spectra(interpolated, new_wavelength_range)

            # flatten normalized (convert from (1, N) -> (N,))
            spectrum = normalized.flatten()

            if not filter_spectra(new_wavelength_range, spectrum, source_only, emission):
                skipped_count += 1
                continue

            kept_count += 1
            all_spectra.append(spectrum)
            all_labels.append(final_label)

    print(f"\n[Filter Summary] Kept: {kept_count}, Skipped: {skipped_count}, Total: {kept_count + skipped_count}")

    # === Return all processed data without train/test split ===
    X = np.array(all_spectra)
    y = np.array(all_labels)

    return X, y, new_wavelength_range


def evaluate_onnx_model(onnx_model_path, X_test, y_test, label_encoder):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    import onnxruntime as ort

    # Ensure X_test is a DataFrame with named columns
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]
    X_np = X_test.astype(np.float32).to_numpy()

    X_np = X_test.astype(np.float32).to_numpy()

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    y_pred = session.run([output_name], {input_name: X_np})[0]
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int)

    # Encode test labels
    y_test = pd.Series(y_test).apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
    y_test_encoded = label_encoder.transform(y_test)

    # Metrics
    test_accuracy = accuracy_score(y_test_encoded, y_pred_labels)
    test_error = 1 - test_accuracy

    # Try metrics that fail on single class
    try:
        precision = precision_score(y_test_encoded, y_pred_labels, zero_division=0)
        recall = recall_score(y_test_encoded, y_pred_labels, zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred_labels, zero_division=0)
        auc = roc_auc_score(y_test_encoded, y_pred[:, 1]) if y_pred.ndim > 1 and len(np.unique(y_test_encoded)) == 2 else None
    except ValueError:
        precision = recall = f1 = auc = None

    # Safe confusion matrix for binary setup
    cm = confusion_matrix(y_test_encoded, y_pred_labels, labels=[0, 1])
    tn = fp = fn = tp = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        if y_test_encoded[0] == 0:
            tn = cm[0][0]
        else:
            tp = cm[0][0]

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n📊 ONNX Test Set Evaluation")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Error: {test_error:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}" if recall is not None else "Sensitivity: N/A")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}" if precision is not None else "Precision: N/A")
    print(f"F1 Score: {f1:.4f}" if f1 is not None else "F1 Score: N/A")
    print(f"AUC: {auc:.4f}" if auc is not None else "AUC: N/A")

    return {
        "test_accuracy": test_accuracy,
        "test_error": test_error,
        "precision": precision,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "f1_score": f1,
        "auc": auc
    }