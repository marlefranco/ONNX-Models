
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

        function [value, summary] = readCSV( dsLocation, darkReference, options )
            % Input validation
            arguments
                dsLocation (1,:) string = ""
                darkReference (1,1) string = ""
                options.FileIndex (1,1) double = inf
                options.SpectrometerType (1,1) {mustBeMember(options.SpectrometerType, ["XL", "CL"])} = "XL"
                options.IncludeSubfolders (1,1) logical = true
            end
            % 
            % % 1) Gather all possible *.csv files (manual extension filter)
            % dsFileSetObj = matlab.io.datastore.DsFileSet(dsLocation, ...
            %     "IncludeSubfolders", options.IncludeSubfolders);
            % 
            % fileInfo = dsFileSetObj.resolve;
            % files = fileInfo.FileName;

             % Read dark reference
            % Specify the range for dark reference data
            darkReferenceRange = 'B7:CW2074'; % Columns B to CW contain the 100 dark reference samples
            % Read dark reference intensity data
            darkIntensities = readmatrix(darkReference, 'Range', darkReferenceRange);
            
            % Compute the average across the 100 columns
            darkReferenceAvg = mean(darkIntensities, 2); % Compute the mean for each row (wavelength)

            darkReferenceAvg = darkReferenceAvg';

            
                   
                   % Get all files present in location
            files = matlab.io.datastore.DsFileSet(fullfile(dsLocation), ...
                "FileExtensions", ".csv", ...
                "IncludeSubfolders", options.IncludeSubfolders).resolve.FileName;
        
            if isempty(files)
                disp("No files to read.");
                value = [];
                summary = [];
                return;
            end
        
                 
            % What files to read?
            if options.FileIndex == inf
                fileIndex = 1:length(files);
            else
                fileIndex = options.FileIndex;
            end

        
           % allTables = cell(1,numel(fileIdx));  % store partial data for each file
             % Initialize data table
            value = cell(1, length(fileIndex));
            % 3) Loop over each file
             for currFileIdx = fileIndex(:)'
                 try
                    fn = files(currFileIdx);


                    %% Read in spectroscopy data
                    %% Read spectroscopy data
                    data = readtable(fn, 'Range', 'A5:EU2074');
                    wavelength = rows2vars(data(3:end, 1));
                    wavelength(:, 1) = []; 
                    wavelength = wavelength(:, 1:2048);
                    data(:,1) = []; % remove wavelength column
                    data = rows2vars(data);

                    data(:,1) = [];

                    Rotation = data{:,1};
                    data(:,1) = [];

                    % consider only first 2048 readings for uniformity
                    position = data{:,1};
                    data(:,1) = [];
                    data = data(:,1:2048);

                    data.Properties.VariableNames = ...
                        "Feature " + string(1:width(data));

                    % Subtract dark reference from each row in data
                    dataSubtracted = data{:,:} - darkReferenceAvg(:,1:2048); % Perform row-wise subtraction

                    
                    
                    % % Convert back to table (optional, if you want to maintain the table structure)
                    % dataSubtractedTable = array2table(dataSubtracted, ...
                    %     'VariableNames', data.Properties.VariableNames);
                    % 
                    % % Assign the updated table back to the original variable (optional)
                    % data = dataSubtractedTable;

                    % Apply Savitzky-Golay smoothing along features (columns)
                    %intensitiesSmooth = smoothdata(dataSubtracted, 2, 'sgolay', 15);

                    %low pass filter
                    % fs = 100;  % Sampling rate (assumed)
                    % cutoffFreq = 20;  % Cutoff frequency (in Hz)
                    % order =20;  % Filter order
                    % [b, a] = butter(order, cutoffFreq / (fs/2), 'low');
                    % intensitiesSmooth = filtfilt(b, a, dataSubtracted);
                    % % Move Mean Settings 
                   % windowSize = 35; % Define window size (3-point moving average)
                   % intensitiesSmooth = movmean(dataSubtracted, windowSize, 2);
                    numtaps = 35; % Equivalent to Python's numtaps
                    filterOrder = numtaps - 1; % FIR filter order
                    cutoffFreq = 0.1; % Normalized cutoff frequency
                    
                    % Design FIR filter using Hamming window
                    firCoeffs = fir1(filterOrder, cutoffFreq, 'low', hamming(numtaps));
                    
                    % Apply zero-phase filtering to avoid phase distortion
                    intensitiesSmooth = filtfilt(firCoeffs, 1, dataSubtracted);
                    %% Median Filtering
                    % windowSize1 = 35;  % Define window size
                    % intensitiesSmooth = medfilt1(dataSubtracted, windowSize1);
                    % Parameters for the Gaussian filter
                   %  windowSize = 35; % Define the window size
                   %  sigma = 10;      % Standard deviation of the Gaussian kernel
                   % 
                   %  % Create the Gaussian kernel
                   %  gaussianKernel = gausswin(windowSize, sigma);
                   % 
                   %  % Normalize the kernel to ensure the sum is 1
                   %  gaussianKernel = gaussianKernel / sum(gaussianKernel);
                   % 
                   % % Initialize output matrix
                   %  intensitiesSmooth = zeros(size(dataSubtracted));
                   % 
                   %  % Apply Gaussian filter row-wise
                   %  for i = 1:size(dataSubtracted, 1)
                   %      intensitiesSmooth(i, :) = conv(dataSubtracted(i, :), gaussianKernel, 'same');
                   %  end
                                                            
                   % Combine Rotation, Position, and Intensities into one table
                    combinedData = table(Rotation, position, intensitiesSmooth);
                    
                    % Combine Rotation and Position as unique pairs
                    uniquePairs = unique([combinedData.Rotation, combinedData.position], 'rows', 'stable');
                    
                    % Find group indices for each (Rotation, Position) combination
                    [G, ~] = findgroups(combinedData.Rotation, combinedData.position);
                    % Initialize an array to store the average spectra for each (Rotation, Position)
                    nFeatures = size(intensitiesSmooth, 2); % Number of features (columns in intensitiesSmooth)
                    avgSpectra = nan(max(G), nFeatures); % Use max(G) instead of length(uniquePairs)
                    
                    % Calculate the mean spectrum for each unique group
                    for i = 1:size(uniquePairs, 1)
                        % Extract indices for the current group
                        groupIndices = (G == i);
                        
                        % Extract the corresponding spectra for this group
                        spectraGroup = intensitiesSmooth(groupIndices, :);
                        
                        % Compute the mean spectrum for the group
                        avgSpectra(i, :) = mean(spectraGroup, 1, 'omitnan');
                    end

                   wavelength = wavelength{1, :}; 
                   % Extract the numeric array from the table
                   intensity = avgSpectra(:, 1:2048);  
                   new_wavelength_range = linspace(400, 940, 2048);
                    % Initialize a matrix to store interpolated intensities
                    interpolated_intensity = zeros(size(intensity));
                    
                    % Interpolate each row separately
                    for i = 1:size(intensity, 1)
                        interpolated_intensity(i, :) = interp1(wavelength, intensity(i, :), new_wavelength_range, 'linear', 'extrap');
                    end

                    % Preallocate arrays
                    largest_negative = zeros(size(interpolated_intensity, 1), 1);
                    offset_corrected = zeros(size(interpolated_intensity));
                    normalization_factor = zeros(size(interpolated_intensity, 1), 1);
                    normalized_spectra = zeros(size(interpolated_intensity));

                    for j = 1:size(interpolated_intensity, 1)
                       % Offset correction: Offset the waveform by 0.1 + largest negative value
                        largest_negative(j) = min(interpolated_intensity(j, :));
                        offset_corrected(j, :) = interpolated_intensity(j, :) + (0.1 - largest_negative(j)); 

                       % Find the value corresponding to 630 nm
                        differences = abs(wavelength - 630); 
                        [~, min_index] = min(differences); % Get the index of the closest wavelength
                        normalization_factor(j) = offset_corrected(j, min_index); 

                       % Normalize the offset-corrected dataset
                        normalized_spectra(j, :) = offset_corrected(j, :) / normalization_factor(j); 
                    end
                    
                   data = array2table(normalized_spectra);
                    data.Properties.VariableNames = ...
                        "Feature " + string(1:width(data));
                    % assign response data
                    data.Rotation = uniquePairs(:, 1); % Add Rotation
                    data.Position = uniquePairs(:, 2); % Add Position
                    %% Assign metadata
                     % Extract metadata from rows 2081 to 2138
                    opts = delimitedTextImportOptions('NumVariables', 2, ...
                        'DataLines', [2081 2138], ...
                        'VariableNamesLine', 0, ...
                        'VariableTypes', ["string", "string"], ...
                        'ExtraColumnsRule', 'ignore');
                    metaData = readtable(fn, opts);
                    metaData = rows2vars(metaData, ...
                        'VariableNamingRule', 'preserve', ...
                        'VariableNamesSource', 1);


                    % Extract light source
                    lightSource = metaData.("Light Source Type"){1};
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
                catch
                    fprintf('Unable to read from file: %s', fn);
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


% classdef import
%     %IMPORT Import datastore
%     %
%     % Methods:
%     %   ops.import.read - Import OSTA data
%     %
%     %
% 
%     methods (Static)
% 
%         function [value, summary] = read( dslocation, options )
% 
%             arguments
%                 dslocation (1,1) string = ""
%                 options.Class (1,1) string {mustBeMember(options.Class, ["All", "ABC", "D"])} = "All"
%             end
% 
%             if ~isfolder( dslocation ) && ~isfile( dslocation )
%                 error("Location of datastore not found.")
%             end
% 
%             files = matlab.io.datastore.DsFileSet( dslocation, ...
%                 "FileExtensions", ".xlsx").resolve.FileName;
% 
%             switch options.Class
%                 case "ABC"
%                     tF = contains(files, "all_data");
%                     files = files(tF);
%                 case "D"
%                     tF = contains(files, "Class_D");
%                     files = files(tF);
%                 otherwise
%                     % continue
%             end
% 
%             ds = spreadsheetDatastore( files, ...
%                 "FileExtensions", ".xlsx", ...
%                 "IncludeSubfolders", true, ...
%                 "TextType", "string", ...
%                 "ReadVariableNames", true, ...
%                 "VariableNamingRule", "preserve");
% 
%             data = readall(ds);
% 
%             data.Feature = [];
%             tF = ismember(data.Properties.VariableNames, ["Sample ID", "Response"]);
%             data.Properties.VariableNames(~tF) = "Feature " + data.Properties.VariableNames(~tF);
% 
%             data.Response = categorical(data.Response);
% 
%             value = data;
% 
%             summary = groupsummary( value, "Response" );
% 
%         end
% 
%         %function [value, summary] = readCSV( dsLocation, white_location, options)
%         function [value, summary] = readCSV( dsLocation, options)
%             % Input validation
%             arguments
%                 dsLocation (1,:) string = ""; 
%                 %white_location (1,:) string = "";
%                 options.FileIndex (1,1) double = inf;
%                 options.SpectrometerType (1,1) ...
%                     {mustBeMember(options.SpectrometerType, ["XL", "CL"])} = ...
%                     "XL";
%                 options.IncludeSubfolders (1,1) logical = true;
% 
% 
%             end
% 
%             % Get all files present in location
%             files = matlab.io.datastore.DsFileSet( fullfile(dsLocation), ...
%                 "FileExtensions", ".csv", ...
%                 "IncludeSubfolders", options.IncludeSubfolders).resolve.FileName;
% 
%             if isempty(files)
%                 disp("No files to read.")
%                 value = [];
%                 summary = [];
%                 return;
%             end
% 
%             if strcmp(options.SpectrometerType, "CL")
%                 files = files(contains(files, "CL"));
%             elseif strcmp(options.SpectrometerType, "XL")
%                 files = files(contains(files, "XL"));
% 
%             end
% 
%             if isempty(files)
%                 fprintf("No files to read. You have selected %s " + ...
%                     "spectrometer type. Try changing it to %s type.", ...
%                     options.SpectrometerType, ...
%                     setdiff(["XL", "CL"], options.SpectrometerType));
%                 value = [];
%                 summary = [];
%                 return;
%             end
% 
%             % What files to read?
%             if options.FileIndex==inf
%                 fileIndex = 1:length(files);
%             else
%                 fileIndex = options.FileIndex;
%             end
% 
%             % Initialize data table
%             % Meta data columns
%             value = cell(1, length(fileIndex));
%             % Loop through every file
%             for currFileIdx = fileIndex(:)'
%                 try
%                     fn = files(currFileIdx);
% 
%                     %% Read in spectroscopy data
%                     data = readtable( fn , ...
%                         'Range', 'A60');
% 
%                     %Initialize wavelength
%                     wavelength = rows2vars(data(3: end, 1));
%                     wavelength(:,1) = [];
%                     wavelength = wavelength(:,1:2048);
% 
% 
%                     data(:,1) = []; 
%                     data = rows2vars(data);
% 
%                     data(:,1) = [];
%                     RotationCount = data{:,1}; 
%                     data(:,1) = []; 
% 
%                     %data(:,1) = [];
%                     responseRegression = data{:,1};
%                     data(:,1) = [];
% 
%                     % consider only first 2048 readings for uniformity
%                     data = data(:,1:2048);
%                     % % %truncate without rounding
%                     % data = table2array(data); 
%                     % data = fix(data * 10^1)/10^1; 
%                     % data = array2table(data); 
% 
%                     intensity = data; 
%                     wavelength = wavelength{1,:}; 
%                     intensity = intensity{:,:};
% 
% 
%                     new_wavelength_range = linspace(400, 940, 2048);
%                     interpolated_intensity = zeros(size(intensity));
%                     for i = 1:size(intensity, 1)
%                         %interpolated_intensity(i, :) = spline(wavelength, intensity(i, :), new_wavelength_range);
%                         interpolated_intensity(i, :) = interp1(wavelength, intensity(i,:), new_wavelength_range, 'linear', 'extrap');
%                     end
%                     % smooth_data = movmean(interpolated_intensity, 6, 2); 
%                     % data = smooth_data; 
%                     save('svm_interpolated_spectral_data.mat', 'interpolated_intensity', 'new_wavelength_range');
%                     % largest_negative = zeros(size(interpolated_intensity, 2), 1); 
%                     % offset_white_reference = zeros(size(interpolated_intensity)); 
%                     % normalization_factor = zeros(size(interpolated_intensity, 2), 1); 
%                     % normalized_data = zeros(size(interpolated_intensity)); 
%                     % correction_coef = zeros(size(interpolated_intensity)); 
%                     % new_spectra = zeros(size(interpolated_intensity)); 
%                     % % table = readtable(white_location); 
%                     % % alphaSpectra = table.Alpha1_Spectrometer1(1:2028, :);
%                     % 
%                    %  for j = 1:size(interpolated_intensity, 1)
%                    %      %offset the waveform by 0.1 + largest negative value
%                    %      largest_negative(j) = min(interpolated_intensity(j,:)); 
%                    %      offset_white_reference(j,:) = interpolated_intensity(j,:) + (0.1 + largest_negative(j)); 
%                    % 
%                    %      %find value corresponding to 730 nm
%                    %      differences = abs(wavelength - 630); 
%                    %      [min_difference, min_index] = min(differences); 
%                    %      closest_wavelength = wavelength(min_index); 
%                    %      normalization_factor(j) = interpolated_intensity(j, min_index); 
%                    % 
%                    %      %normalize dataset 
%                    %      new_spectra(j,:) = interpolated_intensity(j,:)/normalization_factor(j); 
%                    % 
%                    %  % %     %Experiment: Multiply normalization by a factor of
%                    %  % %     %10000
%                    %  % %     % new_spectra(j, :) = normalized_data(j, :) .* 10000; 
%                    %  % % 
%                    %  % %     %correction coeffients
%                    %  % %     % correction_coef = normalized_data(j,:) ./ alphaSpectra';
%                    %  % %     % new_spectra(j,:) = interpolated_intensity(j,:) .* correction_coef;
%                    % end 
% 
% 
%                     % %Read in the white reference and dark reference table
%                     % white_dark_values = readtable(white_location);
%                     % white_dark_values = white_dark_values(1:2028, :); 
%                     % %get the white reference - only for white ab off for now
%                     % white_reference = white_dark_values.whiteAbOff1Ms - white_dark_values.dark1Ms; 
%                     % 
%                     % %offset the waveform by 0.1 + largest negative value
%                     % largest_negative = max(white_reference < 0); 
%                     % offset_white_reference = white_reference + (0.1 + largest_negative); 
%                     % 
%                     % %find value corresponding to 730 nm
%                     % differences = abs(white_dark_values.Var1 - 730); 
%                     % [min_difference, min_index] = min(differences); 
%                     % closest_wavelength = white_dark_values.Var1(min_index); 
%                     % normalization_factor = offset_white_reference(min_index); 
%                     % 
%                     % %normalize dataset 
%                     % normalized_data = offset_white_reference/normalization_factor; 
%                     % 
%                     % %correction coefficients
%                     % correction_coef = normalized_data ./ white_dark_values.Alpha1_Spectrometer1; 
%                     % new_spectra = interpolated_intensity .* correction_coef'; 
% 
%                     % Create a new table with interpolated spectral values
% 
%                     % Clipping the data to the range 430 nm to 730 nm
%                     % clip_start = 430;
%                     % clip_end = 730;
%                     % 
%                     % % Find indices for clipping range
%                     % [~, clip_start_idx] = min(abs(new_wavelength_range - clip_start));
%                     % [~, clip_end_idx] = min(abs(new_wavelength_range - clip_end));
%                     % 
%                     % % Clip the interpolated intensity data
%                     % clipped_intensity = interpolated_intensity(:, clip_start_idx:clip_end_idx);
%                     % clipped_wavelength_range = new_wavelength_range(clip_start_idx:clip_end_idx);
%                     % 
%                     % for j = 1:size(clipped_intensity, 1)
%                     %     %offset the waveform by 0.1 + largest negative value
%                     %     largest_negative(j) = min(clipped_intensity(j,:)); 
%                     %     offset_white_reference(j,:) = clipped_intensity(j,:) + (0.1 + largest_negative(j)); 
%                     % 
%                     %     %find value corresponding to 730 nm
%                     %     differences = abs(clipped_wavelength_range - 630); 
%                     %     [min_difference, min_index] = min(differences); 
%                     %     closest_wavelength = clipped_wavelength_range(min_index); 
%                     %     normalization_factor(j) = clipped_intensity(j, min_index); 
%                     % 
%                     %     %normalize dataset 
%                     %     new_spectra(j,:) = clipped_intensity(j,:)/normalization_factor(j); 
%                     % 
%                     % 
%                     % % 
%                     % %     %correction coeffients
%                     % %     % correction_coef = normalized_data(j,:) ./ alphaSpectra';
%                     % %     % new_spectra(j,:) = interpolated_intensity(j,:) .* correction_coef;
%                     % end 
% 
%                     data = array2table(interpolated_intensity);
%                     data.Properties.VariableNames = "Feature" + string(1:width(data)); 
% 
%                     % data.Properties.VariableNames = ...
%                     %     "Feature " + string(1:width(data));
% 
%                     % assign response data
%                    %data.ResponseRegression = round(responseRegression);
%                    data.ResponseRegression = round(responseRegression);
%                    data.Rotation = round(RotationCount); 
% 
%                     %% Assign metadata
% 
%                     opts = delimitedTextImportOptions('NumVariables', 2, ...
%                         'DataLines', [1 58], ...
%                         'VariableNamesLine', 0, ...
%                         'VariableTypes', ["string", "string"], ...
%                         'ExtraColumnsRule', 'ignore' );
%                     metaData = readtable(fn, opts);
%                     metaData = rows2vars(metaData, ...
%                         'VariableNamingRule', 'preserve', ...
%                         'VariableNamesSource', 1);
% 
%                     % Extract light source
%                     lightSource = metaData.("Light Source"){1};
%                     data.LightSource = repmat(string(lightSource), height(data), 1);
% 
%                     % Extract fiber type
%                     probeSize = metaData.("Fiber Type"){1};
%                     data.ProbeSize = repmat(string(probeSize), height(data), 1);
% 
%                     % Extract aiming beam status
%                     aimingBeam = metaData.("Aiming Beam Status"){1};
%                     data.AimingBeam =repmat(string(aimingBeam), height(data), 1);
% 
%                     % Extract stone info
%                     targetType = metaData.("Target Type"){1};
%                     data.TargetType = repmat(string(targetType), height(data), 1);
% 
%                     % Extract stone info
%                     targetNumber = metaData.("Target Number"){1};
%                     data.TargetNumber = repmat(string(targetNumber), height(data), 1);
% 
%                     % Extract integration time used
%                     integrationTimeUsed = metaData.("Integration Time Used (mS)"){1};
%                     data.IntegrationTimeUsed = repmat(string(integrationTimeUsed), height(data), 1);
% 
%                      % Extract integration time used
%                     SpectrometerUsed = metaData.("Spec Used"){1};
%                     data.SpectrometerUsed = repmat(string(SpectrometerUsed), height(data), 1);
% 
%                     % Extract filename
%                     data.FileName = repmat(fn, height(data), 1);
% 
%                     value{ismember(fileIndex, currFileIdx)} = data;
%                 catch
%                     fprintf('Unable to read from file: %s', fn);
%                     continue;
%                 end
%             end
%             % for currFileIdx = fileIndex(:)'
%             %     try
%             %         fn = files(currFileIdx);
%             % 
%             %         %% Read in spectroscopy data
%             %         data = readtable(fn, 'Range', 'A60');
%             % 
%             %         % Initialize wavelength
%             %         wavelength = rows2vars(data(3:end, 1));
%             %         wavelength(:, 1) = [];
%             %         wavelength = wavelength(:, 1:2048);
%             % 
%             %         data(:, 1) = [];
%             %         data = rows2vars(data);
%             % 
%             %         data(:, 1) = [];
%             %         RotationCount = data{:, 1};
%             %         data(:, 1) = [];
%             % 
%             %         responseRegression = data{:, 1};
%             %         data(:, 1) = [];
%             % 
%             %         % Consider only first 2048 readings for uniformity
%             %         data = data(:, 1:2048);
%             % 
%             %         intensity = data;
%             %         wavelength = wavelength{1, :};
%             %         intensity = intensity{:,:};
%             % 
%             %         new_wavelength_range = linspace(400, 940, 2048);
%             %         interpolated_intensity = zeros(size(intensity));
%             %         for i = 1:size(intensity, 1)
%             %             interpolated_intensity(i, :) = interp1(wavelength, intensity(i, :), new_wavelength_range, 'linear', 'extrap');
%             %         end
%             % 
%             %         save('interpolated_spectral_data.mat', 'interpolated_intensity', 'new_wavelength_range');
%             % 
%             %         data = array2table(interpolated_intensity);
%             %         data.Properties.VariableNames = "Feature" + string(1:width(data)); 
%             % 
%             %         data.ResponseRegression = round(responseRegression);
%             %         data.Rotation = round(RotationCount); 
%             % 
%             %         %% Assign metadata
%             %         opts = delimitedTextImportOptions('NumVariables', 2, ...
%             %             'DataLines', [1 58], ...
%             %             'VariableNamesLine', 0, ...
%             %             'VariableTypes', ["string", "string"], ...
%             %             'ExtraColumnsRule', 'ignore');
%             %         metaData = readtable(fn, opts);
%             %         metaData = rows2vars(metaData, ...
%             %             'VariableNamingRule', 'preserve', ...
%             %             'VariableNamesSource', 1);
%             % 
%             %         % Extract light source
%             %         lightSource = metaData.("Light Source"){1};
%             %         data.LightSource = repmat(string(lightSource), height(data), 1);
%             % 
%             %         % Extract fiber type
%             %         probeSize = metaData.("Fiber Type"){1};
%             %         data.ProbeSize = repmat(string(probeSize), height(data), 1);
%             % 
%             %         % Extract aiming beam status
%             %         aimingBeam = metaData.("Aiming Beam Status"){1};
%             %         data.AimingBeam = repmat(string(aimingBeam), height(data), 1);
%             % 
%             %         % Extract stone info
%             %         targetType = metaData.("Target Type"){1};
%             %         data.TargetType = repmat(string(targetType), height(data), 1);
%             % 
%             %         % Extract stone info
%             %         targetNumber = metaData.("Target Number"){1};
%             %         data.TargetNumber = repmat(string(targetNumber), height(data), 1);
%             % 
%             %         % Extract integration time used
%             %         integrationTimeUsed = metaData.("Integration Time Used (mS)"){1};
%             %         data.IntegrationTimeUsed = repmat(string(integrationTimeUsed), height(data), 1);
%             % 
%             %         % Extract spectrometer used
%             %         SpectrometerUsed = metaData.("Spec Used"){1};
%             %         data.SpectrometerUsed = repmat(string(SpectrometerUsed), height(data), 1);
%             % 
%             %         % Extract filename
%             %         data.FileName = repmat(fn, height(data), 1);
%             % 
%             %         % Initialize value
%             %         value{ismember(fileIndex, currFileIdx)} = data;
%             % 
%             %     %     %% Duplicate data with phase shifts
%             %     %     delta_shifts = [0.1, 0.3, 0.5, 1, 5, 7, 10]; % Define the shifts in nm
%             %     % 
%             %     %     for delta = delta_shifts
%             %     %         % Positive shifts
%             %     %         shifted_wavelength_pos = wavelength + delta; % Shift wavelengths positively
%             %     %         shifted_intensity_pos = zeros(size(intensity));
%             %     % 
%             %     %         for i = 1:size(intensity, 1)
%             %     %             shifted_intensity_pos(i, :) = interp1(wavelength, intensity(i, :), shifted_wavelength_pos, 'linear', 'extrap');
%             %     %         end
%             %     % 
%             %     %         % Normalize the interpolated data at 630nm
%             %     %         normalized_shifted_intensity_pos = normalizeByValueAt630nm(shifted_intensity_pos, new_wavelength_range);
%             %     % 
%             %     %         shifted_data_pos = array2table(normalized_shifted_intensity_pos);
%             %     %         shifted_data_pos.Properties.VariableNames = "Feature" + string(1:width(shifted_data_pos));
%             %     % 
%             %     %         shifted_data_pos.ResponseRegression = round(responseRegression);
%             %     %         shifted_data_pos.Rotation = round(RotationCount);
%             %     % 
%             %     %         % Append metadata
%             %     %         shifted_data_pos.LightSource = repmat(string(lightSource), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.ProbeSize = repmat(string(probeSize), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.AimingBeam = repmat(string(aimingBeam), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.TargetType = repmat(string(targetType), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.TargetNumber = repmat(string(targetNumber), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.IntegrationTimeUsed = repmat(string(integrationTimeUsed), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.SpectrometerUsed = repmat(string(SpectrometerUsed), height(shifted_data_pos), 1);
%             %     %         shifted_data_pos.FileName = repmat(fn, height(shifted_data_pos), 1);
%             %     % 
%             %     %         % Append positive shifted data to value
%             %     %         value{ismember(fileIndex, currFileIdx)} = [value{ismember(fileIndex, currFileIdx)}; shifted_data_pos];
%             %     %     end
%             %     % 
%             %     %     save('phaseshiftDuplicate_spectraldata.mat', 'shifted_data_pos', 'new_wavelength_range');
%             %     % catch ME
%             %     %     fprintf('Unable to read from file: %s. Error: %s\n', fn, ME.message);
%             %     %     continue;
%             %     % end
%             % 
%             %     % delta_shifts = [0.1, 0.3, 0.5, 1, 5, 10]; % Define the shifts in nm
%             %     % 
%             %     % for delta = delta_shifts
%             %     %     % Negative shifts
%             %     %     shifted_wavelength_neg = wavelength - delta; % Shift wavelengths negatively
%             %     %     shifted_intensity_neg = zeros(size(intensity));
%             %     % 
%             %     %     for i = 1:size(intensity, 1)
%             %     %         shifted_intensity_neg(i, :) = interp1(wavelength, intensity(i, :), shifted_wavelength_neg, 'linear', 'extrap');
%             %     %     end
%             %     % 
%             %     %     % Normalize the interpolated data at 630nm
%             %     %     normalized_shifted_intensity_neg = normalizeByValueAt630nm(shifted_intensity_neg, new_wavelength_range);
%             %     % 
%             %     %     shifted_data_neg = array2table(normalized_shifted_intensity_neg);
%             %     %     shifted_data_neg.Properties.VariableNames = "Feature" + string(1:width(shifted_data_neg));
%             %     % 
%             %     %     shifted_data_neg.ResponseRegression = round(responseRegression);
%             %     %     shifted_data_neg.Rotation = round(RotationCount);
%             %     % 
%             %     %     % Append metadata
%             %     %     shifted_data_neg.LightSource = repmat(string(lightSource), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.ProbeSize = repmat(string(probeSize), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.AimingBeam = repmat(string(aimingBeam), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.TargetType = repmat(string(targetType), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.TargetNumber = repmat(string(targetNumber), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.IntegrationTimeUsed = repmat(string(integrationTimeUsed), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.SpectrometerUsed = repmat(string(SpectrometerUsed), height(shifted_data_neg), 1);
%             %     %     shifted_data_neg.FileName = repmat(fn, height(shifted_data_neg), 1);
%             %     % 
%             %     %     % Append negative shifted data to value
%             %     %     value{ismember(fileIndex, currFileIdx)} = [value{ismember(fileIndex, currFileIdx)}; shifted_data_neg];
%             %     % end
%             %     % 
%             %     % save('phaseshiftDuplicate_spectraldata_neg.mat', 'shifted_data_neg', 'new_wavelength_range');
%             %     % catch ME
%             %     %     fprintf('Unable to read from file: %s. Error: %s\n', fn, ME.message);
%             %     %     continue;
%             %     % end
%             % end
%             % 
%             % Function to normalize by value at 630nm
%             function normalizedDataArray = normalizeByValueAt630nm(dataArray, wavelength)
%                 % Initialize the normalized data array
%                 normalizedDataArray = zeros(size(dataArray));
% 
%                 % Find the index of the wavelength closest to 630 nm
%                 [~, min_index] = min(abs(wavelength - 630));
% 
%                 % Loop through each spectrum
%                 for j = 1:size(dataArray, 1)
%                     % Get the normalization factor (value at 630 nm for the current spectrum)
%                     normalization_factor = dataArray(j, min_index);
% 
%                     % Check if normalization factor is zero to avoid division by zero
%                     if normalization_factor == 0
%                         warning('Normalization factor for spectrum %d is zero. Skipping normalization.', j);
%                         normalizedDataArray(j, :) = NaN; % or handle appropriately
%                     else
%                         % Normalize the entire spectrum by the value at 630 nm
%                         normalizedDataArray(j, :) = dataArray(j, :) / normalization_factor;
%                     end
% 
%                     % Debugging information (optional)
%                     fprintf('Spectrum %d: Normalization Factor at 630 nm = %.4f\n', j, normalization_factor);
%                     fprintf('Normalized Spectrum %d: Min Value = %.4f, Max Value = %.4f\n', j, min(normalizedDataArray(j, :)), max(normalizedDataArray(j, :)));
%                 end
%             end
% 
%            % end
%             value = vertcat(value{:});
%             metaDataVars = value.Properties.VariableNames(...
%                 ~contains(value.Properties.VariableNames, "Feature"));
%             summary = groupsummary( value, metaDataVars );
%         end
% 
% 
%     end
% 
% end
% 
% function normalizedDataArray = normalizeByValueAt630nm(dataArray, wavelength)
%     % Initialize the normalized data array
%     normalizedDataArray = zeros(size(dataArray));
% 
%     % Find the index of the wavelength closest to 630 nm
%     [~, min_index] = min(abs(wavelength - 630));
%     if isempty(min_index)
%         error('Wavelength 630 nm not found in the given wavelength range.');
%     end
% 
%     % Loop through each spectrum
%     for j = 1:size(dataArray, 1)
%         % Get the normalization factor (value at 630 nm for the current spectrum)
%         normalization_factor = dataArray(j, min_index);
% 
%         % Check if normalization factor is zero to avoid division by zero
%         if normalization_factor == 0
%             warning('Normalization factor for spectrum %d is zero. Skipping normalization.', j);
%             normalizedDataArray(j, :) = NaN; % or handle appropriately
%         else
%             % Normalize the entire spectrum by the value at 630 nm
%             normalizedDataArray(j, :) = dataArray(j, :) / normalization_factor;
%         end
% 
%         % % Debugging information (optional)
%         % fprintf('Spectrum %d: Normalization Factor at 630 nm = %.4f\n', j, normalization_factor);
%         % fprintf('Normalized Spectrum %d: Min Value = %.4f, Max Value = %.4f\n', j, min(normalizedDataArray(j, :)), max(normalizedDataArray(j, :)));
%     end
% end
% 
% 
