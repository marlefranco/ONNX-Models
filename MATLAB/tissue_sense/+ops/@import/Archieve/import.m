classdef import
    %IMPORT Import datastore
    %
    % Methods:
    %   ops.import.read - Import OSTA data
    %
    %

    methods (Static)

        function [value, summary] = read( dslocation, options )

            arguments
                dslocation (1,1) string = ""
                options.Class (1,1) string {mustBeMember(options.Class, ["All", "ABC", "D"])} = "All"
            end

            if ~isfolder( dslocation ) && ~isfile( dslocation )
                error("Location of datastore not found.")
            end

            files = matlab.io.datastore.DsFileSet( dslocation, ...
                "FileExtensions", ".xlsx").resolve.FileName;

            switch options.Class
                case "ABC"
                    tF = contains(files, "all_data");
                    files = files(tF);
                case "D"
                    tF = contains(files, "Class_D");
                    files = files(tF);
                otherwise
                    % continue
            end

            ds = spreadsheetDatastore( files, ...
                "FileExtensions", ".xlsx", ...
                "IncludeSubfolders", true, ...
                "TextType", "string", ...
                "ReadVariableNames", true, ...
                "VariableNamingRule", "preserve");

            data = readall(ds);

            data.Feature = [];
            tF = ismember(data.Properties.VariableNames, ["Sample ID", "Response"]);
            data.Properties.VariableNames(~tF) = "Feature " + data.Properties.VariableNames(~tF);

            data.Response = categorical(data.Response);

            value = data;

            summary = groupsummary( value, "Response" );

        end

        function [value, summary] = readCSV( dsLocation, options )
            % Input validation
            arguments
                dsLocation (1,:) string = "";
                options.FileIndex (1,1) double = inf;
                options.SpectrometerType (1,1) ...
                    {mustBeMember(options.SpectrometerType, ["XL", "CL"])} = ...
                    "XL";
                options.IncludeSubfolders (1,1) logical = false;
            end

            % Get all files present in location
            files = matlab.io.datastore.DsFileSet( fullfile(dsLocation), ...
                "FileExtensions", ".csv", ...
                "IncludeSubfolders", options.IncludeSubfolders).resolve.FileName;

            if isempty(files)
                disp("No files to read.")
                value = [];
                summary = [];
                return;
            end

            if strcmp(options.SpectrometerType, "CL")
                files = files(contains(files, "CL"));
            elseif strcmp(options.SpectrometerType, "XL")
                files = files(contains(files, "XL"));
            end

            if isempty(files)
                fprintf("No files to read. You have selected %s " + ...
                    "spectrometer type. Try changing it to %s type.", ...
                    options.SpectrometerType, ...
                    setdiff(["XL", "CL"], options.SpectrometerType));
                value = [];
                summary = [];
                return;
            end

            % What files to read?
            if options.FileIndex==inf
                fileIndex = 1:length(files);
            else
                fileIndex = options.FileIndex;
            end

            % Initialize data table
            % Meta data columns
            value = cell(1, length(fileIndex));
            % Loop through every file
            for currFileIdx = fileIndex(:)'
                try
                    fn = files(currFileIdx);

                    %% Read in spectroscopy data
                    data = readtable( fn , ...
                        'Range', 'A60');
                    wavelength = rows2vars(data(3:end, 1)); 
                    wavelength(:, 1) = []; 
                    wavelength = wavelength(:, 1:2048); 

                    data(:, 1) = [];
                    data = rows2vars(data);

                    data(:,1) = [];
                    RotationCount = data{:, 1}; 
                    data(:, 1) = []; 

                    responseRegression = data{:,1};
                    data(:,1) = [];
                    

                      %consider only first 2048 readings for uniformity
                    data = data(:, 1:2048); 
                    intensity = data;
                    wavelength = wavelength{1, :};  % Extract the numeric array from the table
                    intensity = intensity{:,:}; 
                    
                   new_wavelength_range = linspace(400, 940, 2048);
                    % Initialize a matrix to store interpolated intensities
                    interpolated_intensity = zeros(size(intensity));
                    
                    % Interpolate each row separately
                    for i = 1:size(intensity, 1)
                        interpolated_intensity(i, :) = interp1(wavelength, intensity(i, :), new_wavelength_range, 'linear', 'extrap');
                    end
                    
                    
                    % % Preallocate cell array to store GPR models for each target
                    % finalGPRModels = cell(size(intensity, 1), 1);
                    % 
                    % rng(42);
                    % for i = 1:size(intensity, 1)
                    %     % Train GPR model with current hyperparameters
                    %     gpr_model = fitrgp(wavelength', intensity(i, :)', 'KernelFunction', 'squaredexponential', 'Standardize', true);
                    % 
                    %     % Save the trained GPR model for each target
                    %     finalGPRModels{i} = gpr_model;
                    % end
                    % 
                    % % Initialize row vector to store interpolated intensities
                    % interpolated_intensities = zeros(1, numel(new_wavelength_range));
                    % 
                    % % Iterate over all targets and accumulate interpolated intensities
                    % for m = 1:size(intensity, 1)
                    %     % Predict using the final GPR model for each target
                    %     interpolated_intensities(m, :) = predict(finalGPRModels{m}, new_wavelength_range(:));
                    % end

                    % 
                    % % Initialize variables to store results
                    % bestRMSE = Inf;
                    % bestKernelParams = [];
                    
                    % % Define the hyperparameters to search over
                    % % kernelParams1 = logspace(-2, 2, 5);  % Range for the first hyperparameter
                    % % kernelParams2 = logspace(-2, 2, 5);  % Range for the second hyperparameter
                    % % 
                    % % Perform grid search
                    % for i = 1:size(intensity, 1)
                    %     %for j = 1:length(kernelParams1)
                    %        % for k = 1:length(kernelParams2)
                    %             % Train GPR model with current hyperparameters
                    %             %currentKernelParams = [kernelParams1(j); kernelParams2(k)];
                    %             gpr_model = fitrgp(wavelength', intensity(i, :)', 'KernelFunction', 'squaredexponential', 'Standardize', true);
                    % 
                    %             % Predict using the GPR model
                    %             predicted_intensities = predict(gpr_model, wavelength(:));
                    % 
                    %        %      % Compute RMSE
                    %        %      currentRMSE = sqrt(mean((intensity(i, :)' - predicted_intensities).^2));
                    %        % 
                    %        %      % Update best hyperparameters if current result is better
                    %        %      if currentRMSE < bestRMSE
                    %        %          bestRMSE = currentRMSE;
                    %        %          bestKernelParams = currentKernelParams;
                    %        %      end
                    %        % % end
                    %     %end
                    % 
                    %     % Train final GPR model with best parameters for the current target
                    %     finalGPRModels{i} = fitrgp(wavelength', intensity(i, :)', 'KernelFunction', 'squaredexponential', 'KernelParameters', bestKernelParams, 'Standardize', true);
                    % end 
                    % 
                    % % Initialize row vector to store interpolated intensities
                    % interpolated_intensities = zeros(1, numel(new_wavelength_range));
                    % 
                    % % Iterate over all targets and accumulate interpolated intensities
                    % for m = 1:size(intensity, 1)
                    %     % Predict using the final GPR model for each target
                    %     interpolated_intensities = interpolated_intensities + predict(finalGPRModels{m}, new_wavelength_range);
                    % end
                    data = array2table(interpolated_intensity); 
                    data.Properties.VariableNames = "Feature" + string(1:width(data)); 


                    % data.Properties.VariableNames = ...
                    %     "Feature " + string(1:width(data));
                    
                  
                    % assign response data
                    data.ResponseRegression = round(responseRegression);
                    data.Rotation = round(RotationCount); 

                    %% Assign metadata
                    opts = delimitedTextImportOptions('NumVariables', 2, ...
                        'DataLines', [1 58], ...
                        'VariableNamesLine', 0, ...
                        'VariableTypes', ["string", "string"], ...
                        'ExtraColumnsRule', 'ignore' );
                    metaData = readtable(fn, opts);
                    metaData = rows2vars(metaData, ...
                        'VariableNamingRule', 'preserve', ...
                        'VariableNamesSource', 1);

                    % Extract light source
                    lightSource = metaData.("Light Source"){1};
                    data.LightSource = repmat(string(lightSource), height(data), 1);

                    % Extract fiber type
                    probeSize = metaData.("Fiber Type"){1};
                    data.ProbeSize = repmat(string(probeSize), height(data), 1);

                    % Extract aiming beam status
                    aimingBeam = metaData.("Aiming Beam Status"){1};
                    data.AimingBeam =repmat(string(aimingBeam), height(data), 1);

                    % Extract stone info
                    targetType = metaData.("Target Type"){1};
                    data.TargetType = repmat(string(targetType), height(data), 1);

                    % Extract stone info
                    targetNumber = metaData.("Target Number"){1};
                    data.TargetNumber = repmat(string(targetNumber), height(data), 1);

                    % Extract integration time used
%                     integrationTimeUsed = metaData.("Integration Time Used (mS)"){1};
                    [~,name,~] = fileparts(fn);
                    integrationTimeUsed = extractBefore(name, "_");
                    data.IntegrationTimeUsed = repmat(integrationTimeUsed, height(data), 1);

                    % Extract spectrometer used
                    spectrometerUsed = metaData.("Spec Used");
                    data.SpectrometerUsed = repmat(string(spectrometerUsed), height(data), 1);

                    % Extract filename
                    data.FileName = repmat(fn, height(data), 1);

                    value{ismember(fileIndex, currFileIdx)} = data;
                catch exception
                        fprintf('Error reading or processing file: %s\n', fn);
                        disp(exception.message);
                        disp(getReport(exception));
                        continue;
                end
            end

            value = vertcat(value{:});
            metaDataVars = value.Properties.VariableNames(...
                ~contains(value.Properties.VariableNames, "Feature"));
            summary = groupsummary( value, metaDataVars );
        end


    end

end

