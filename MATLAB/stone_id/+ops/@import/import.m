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
                    data(:,1) =  [] % remove wavelength column
                    data = rows2vars(data);


                    data(:,1) = [];
                    RotationCount = data{:,1};
                    data(:,1) = [];

                  %  data(:,1) = [];
                    responseRegression = data{:,1};
                    data(:,1) = [];

                    % consider only first 2048 readings for uniformity
                    data = data(:,1:2048);

                    data.Properties.VariableNames = ...
                        "Feature " + string(1:width(data));

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

                    % Rotation = metaData.("Rotation Count of Target"){1};
                    % data.Rotation = repmat(string(Rotation), height(data), 1);

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

