# main.py

import os
from ops.import_trl5 import Import  # Adjust this import based on your module structure
import pandas as pd
from ops.splitn import partition_data_based_on_filenames
from ml_framework.basePipilineTrain import basePipelineTrain
from ml_utility.wavelet_transform import wavelet_transform
from ops.visualize import visualize_data, plot_all_spectra
from ml_framework.classification import train_models, evaluate_model
from ml_utility.anovaTest import anova_significant_peaks, plot_significant_peaks, find_significant_wavelengths, plot_spectral_data_with_roi, find_roi_anova
from ml_utility.pldsa import pls_important_peaks
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.metrics import confusion_matrix, classification_report
import torch
#from ml_utility.cnn_gradcam import compute_grad_cam, preprocess_labels, create_cnn_model
from ml_utility.cnn import create_cnn_model, preprocess_labels, compute_grad_cam, visualize_feature_importance, compute_saliency_map, plot_saliency_map, compute_grad_cam_for_all, plot_all_grad_cams
matplotlib.use('TkAgg')  # Switches to a more stable backend for VS Code
from ml_framework.peakdetection import detect_peaks_across_samples, get_peak_ranges, extract_features_from_ranges, plot_peak_histogram, extract_and_plot_features    # Import functions from peakdetection.py
from ml_framework.spectralFeatures import extract_features, plot_features, plot_spectrum_with_regions
from ml_framework.powerRatioFeatures import calculate_spectral_features, plot_power_ratio_histograms
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def main():
    # Set integra and distanceRange variables
    integra = 1  # Currently fixed to 1 as in the MATLAB code; can be a range or other values
    distanceRange = "ZeroToFour"

    # Define the data location
    ds_location = os.path.join(os.getcwd(), "data", "TRL5-Xenon-AB-OFF")
    #ds_location = os.path.join(os.getcwd(), "data", "Test")
    # Construct the file path
    dark_reference = os.path.join(os.getcwd(), "data", "Dark", "1_S3_Alpha1_DARK AB OFF.xlsx")

    # Initialize options as a dictionary
    options = {
        "FileIndex": float("inf"),  # Read all files
        "SpectrometerType": "XL",  # or "CL"
        "IncludeSubfolders": True  # This needs to be part of the options dictionary
    }

    # Create an instance of Import and call the read_csv method
    data_all, data_summary, wavelength_df = Import.read_csv(ds_location=ds_location, darkReference=dark_reference, options=options, filter_type="fir")
   
    # Print the current timestamp
    print(pd.Timestamp.now().strftime('%d/%m/%y-%H:%M'))

    # Show the first few rows of the data
    print(data_all.head())
    print(data_summary)

    data_all['Response'] = data_all['TargetType']

    # Convert 'integra' to string for filtering
    stringint = str(integra)

    # Apply filters for 'IntegrationTimeUsed' and 'ResponseRegression' in one step
    data = data_all[
        (data_all['IntegrationTimeUsed'] == stringint) & 
        (data_all['Position'] >= 0) & 
        (data_all['Position'] < 5)
    ]

    print(data.head())
    # Replace specific values with "Non-Tissue"
    non_tissue_values = ["Endoscope", "COM", "UA", "BEGO", "Access Sheath", "No Target", "Guidewire", "CHPD", "CYS", "MAGPH", "MAPH"]

    # Check for both variations of "Access Sheath"
    data.loc[data['Response'].str.contains("Access Sheath 13 ?15", regex=True) | data['Response'].isin(non_tissue_values), 'Response'] = "Non-Tissue"
    # Replace "Tissue-Calyx" and "Tissue-Ureter" with "Tissue"
    data.loc[data['Response'].isin(["Tissue-Calyx", "Tissue-Ureter"]), 'Response'] = "Tissue"
    
    # Uncomment the following lines to train stone vs non-stone
    # dataTrain.loc[dataTrain['Response'].isin(["COM", "UA"]), 'Response'] = "Stone"

    # dataTest.loc[dataTest['Response'].isin(["Access Sheath 13/15", "Endoscope", "Tissue-Calyx", "Tissue-Ureter", "BEGO"]), 'Response'] = "Non-Stone"
    # Call the partition function to split data
    train_percent = 80  # Percentage to use for training
    # Call the partition function
    data_train, data_test, train_indices, test_indices = partition_data_based_on_filenames(data, train_percent=80)
    
    print(f"Train size: {len(data_train)}")
    print(f"Test size: {len(data_test)}")

    """ # Get unique combinations for the training set
    unique_train = data_train[['TargetType', 'TargetNumber']].drop_duplicates()
    unique_train['Set'] = 'Train'  # Label the source

    # Get unique combinations for the testing set
    unique_test = data_test[['TargetType', 'TargetNumber']].drop_duplicates()
    unique_test['Set'] = 'Test'  # Label the source

    # Combine the unique combinations from both sets
    combined_unique = pd.concat([unique_train, unique_test], ignore_index=True)

    print("Unique Combinations of TargetType and TargetNumber:")
    print(combined_unique) """
    
    # Keep only predictor columns that contain "Feature" in their names

    predictorNames = [col for col in data_train.columns if "Feature" in col]
    
    X_train = data_train[predictorNames]  # Feature columns only
    Y_train = data_train["Response"]      
    X_test = data_test[predictorNames]    # Feature columns only
    Y_test = data_test["Response"]        # Target column
    
   # plot_all_spectra(X_train, Y_train, wavelength_df)
    
     # Apply Wavelet Transform
   # X_train_transformed = wavelet_transform(X_train)
   # X_test_transformed = wavelet_transform(X_test)

    # Visualize train/test data
   # visualize_data(X_train_transformed, Y_train, X_test_transformed, Y_test)
   
  
    # Reshape data to fit the CNN input (assuming 1D spectral data)
    # ---------------------- Train CNN Model and extract Saliency MAP ----------------------
    wavelengths = wavelength_df # Shape: (num_features, 1)
    wavelengths = np.tile(wavelengths, (X_train.shape[0], 1))  # Shape: (num_samples, num_features)

  #  Combine intensity and wavelengths into one input array
    X_train_combined = np.stack((X_train, wavelengths), axis=-1)  # Shape: (num_samples, num_features, 2)
 
  

   # Preprocess labels (convert 'Tissue' and 'Non-tissue' to 1 and 0)
    Y_train_numeric = preprocess_labels(Y_train)

   # Create the CNN model
    model = create_cnn_model(X_train_combined)

    # Train the model
    history = model.fit(X_train_combined, Y_train_numeric, epochs=10, batch_size=32, validation_split=0.2)
    
    saliency_map = compute_saliency_map(model, X_train_combined, Y_train_numeric)
    plot_saliency_map(saliency_map, wavelengths[0, :],Y_train_numeric, 98)
   
   #-------------------------------------- Grad CAM -----------------------------------------------#
    
    # Compute Grad-CAM for all Tissue and Non-Tissue samples
   # grad_cams = compute_grad_cam_for_all(model, X_train_combined, Y_train_numeric)

    # Plot Grad-CAM results
    #plot_all_grad_cams(grad_cams, wavelengths[0, :])
    # Ensure Model is Called Before Grad-CAM
    # Compute Grad-CAM for Tissue and Non-Tissue
    
    # Ensure Model is Called Before Grad-CAM
# Determine the actual target class dynamically
    # target_class_tissue = int(Y_train_numeric[0])  # First sample's class
    # target_class_non_tissue = int(Y_train_numeric[1])  # Second sample's class

    # # Compute Grad-CAM based on actual class labels
    # tissue_grad_cam = compute_grad_cam(model, X_train_combined[0:1], target_class_tissue)
    # non_tissue_grad_cam = compute_grad_cam(model, X_train_combined[1:2], target_class_non_tissue)
    
    ##### ---------------------------------------- Peak detection and Feature extraction ---------------------------------------------------------------------------------
    #extracted_features = extract_and_plot_features(X_train, Y_train, wavelength_df)
    
  
# Assuming x_train is a DataFrame with rows as spectra and columns as intensity values
    # Assuming wavelength_df is a DataFrame containing the corresponding wavelength values
  #  extracted_features = extract_features(X_train, wavelength_df, Y_train)

    # Display the extracted features
  #  print(extracted_features.head())
   # plot_features(extracted_features)

# Plot the first spectrum with highlighted regions of interest
   # plot_spectrum_with_regions(X_train.iloc[0], wavelength_df)  
   
   
    tissue_mean = X_train[Y_train == "Tissue"].mean(axis=0)
    non_tissue_mean = X_train[Y_train == "Non-Tissue"].mean(axis=0)

    plt.plot(wavelength_df, tissue_mean, label="Tissue", color='green')
    plt.plot(wavelength_df, non_tissue_mean, label="Non-Tissue", color='red')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean Intensity")
    plt.title("Mean Spectra of Tissue vs. Non-Tissue")
    plt.legend()
    plt.show()
    power_ratio_features_train = calculate_spectral_features(X_train, Y_train, wavelength_df)
    
  
    ratio_ranges = {
    "Ratio 1": ("550-680", "490-530"),  # Numerator: 460-490 nm, Denominator: 515-540 nm
    "Ratio 2": ("460-485", "490-530")   # Numerator: 550-680 nm, Denominator: 515-540 nm
    }
    
    ratio_ranges = {
          "Ratio 1": ("460-480", "560-580"), 
          "Ratio 2": ("560-580", "660-680")
        #   "Ratio 1": ("460-490", "515-540"), 
        #   "Ratio 2": ("550-680", "515-540")

      }
    
      

    # Compute power ratio features
    power_ratios_train = calculate_spectral_features(X_train, Y_train, wavelength_df, ratio_ranges)
    # Create a figure with two subplots
    #plot_power_ratio_histograms(power_ratios_train)
    
    power_ratio_features_test = calculate_spectral_features(X_test, Y_test, wavelength_df, ratio_ranges)
    
    # # Step 2: Train the KNN Model on Power Ratio Features
    
    # # Step 2: Extract Features (X) and Labels (y) from Power Ratio DataFrames
    X_train_knn = power_ratios_train.iloc[:, :-1]  # First two columns (features)
    X_train_knn.fillna(X_train_knn.mean(), inplace=True)
    y_train_knn = power_ratios_train.iloc[:, -1]   # Third column (labels)

    X_test_knn = power_ratio_features_test.iloc[:, :-1]  
    X_test_knn.fillna(X_train_knn.mean(), inplace=True)# First two columns (features)
    y_test_knn = power_ratio_features_test.iloc[:, -1]     # Third column (labels)
    print("Before SMOTE:\n", y_train_knn.value_counts())
    
    # smote = SMOTE(random_state=42)
    # X_train_balanced, y_train_balanced = smote.fit_resample(X_train_knn, y_train_knn)  
    # smote_enn = SMOTEENN(random_state=42)
    # X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_knn, y_train_knn) 
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_knn, y_train_knn)
        
    
       
    print("\nAfter SMOTE:\n", pd.Series(y_train_balanced).value_counts())
    
    # Convert resampled dataset to DataFrame
    power_ratios_train_after_smote = pd.DataFrame(X_train_balanced, columns=["Ratio 1", "Ratio 2"])
    power_ratios_train_after_smote["Label"] = y_train_balanced  # Add labels

# Plot histograms after SMOTE
    plot_power_ratio_histograms(power_ratios_train_after_smote)     

    #     # Train KNN Model
    # Train models & get the best one
    best_model, best_cv_results, best_model_name, best_cv_score, label_encoder = train_models(X_train_balanced, y_train_balanced)

# Evaluate the best model
    evaluate_model(best_model, X_train_knn, y_train_knn, X_test_knn, y_test_knn, best_cv_score, best_model_name, label_encoder)    
    
    
    
if __name__ == "__main__":
    main()
