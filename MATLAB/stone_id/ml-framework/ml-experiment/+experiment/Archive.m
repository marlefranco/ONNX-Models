classdef Archive < matlab.mixin.SetGet
    %experiment.Archive Postprocess experiment results
    %
    % Syntax:
    %   evaluation = experiment.Archive( "ModelLocation", modelDS,...
    %        "DataLocation", datalDS );
    %
    %   experiment.Archive methods:
    %
    %        import             - import experiment data
    %        lime               - model interpretation/predictor importance
    %        getmodel           - helper function
    %        confusionchart     - confusion matrix for archived classification model
    %        predictactual      - predict actual plot for archived regression model

    % MathWorks Consulting 2022

    properties
        ModelLocation (1,:) string = ""
        DataLocation (1,:) string = ""
    end

    properties ( GetAccess = 'public', SetAccess = 'private')
        Summary table
        Trials table
    end

    methods
        function obj = Archive( varargin )
            %Postprocess Construct experiment.Archive session

            if ~isempty( varargin )
                set(obj, varargin{:})
            end

        end %function

    end %methods

    methods
        function import(obj)
            %IMPORT Import experiment archive

            % File datastore w/ custom read function to read trials
            fds = fileDatastore( obj.DataLocation, ...
                'ReadFcn', @obj.i_customRead, ...
                'UniformRead', true, ...
                'IncludeSubfolders', true, ...
                'FileExtensions', '.mat');

            obj.Trials = fds.readall();
            obj.Trials = sortrows(obj.Trials, ["Experiment" "Trial"]);

            % Tabular datastore to read model metadata
            sds = spreadsheetDatastore( obj.ModelLocation , ...
                "FileExtensions", ".xlsx", ...
                "IncludeSubfolders", true, ...
                "TextType", "string", ...
                "VariableNamingRule", "preserve");

            this = sds.readall();
            this.Properties.RowNames = this.Row;
            this.Row = [];

            obj.Summary = this;

        end %function

        function lime( obj, trial, mdl, options )
            %LIME Local interpretable model-agnostic explanations (LIME)

            arguments
                obj
                trial (1,1) double {mustBeInteger,mustBePositive}
                mdl (1,1) double {mustBeInteger,mustBePositive}
                options.Type (1,1) string {mustBeMember(options.Type, ...
                    ["classification", "regression"])}
                options.Class (1,1) string = ""
            end

            [mdl, data] = getmodel( obj, trial, mdl );

            if isstruct(mdl)
                mdl = mdl.mdl;
            end

            if ~isempty(obj.Trials) && ~isempty(obj.Summary)

                features    = data.Properties.CustomProperties.Features;
                response    = data.Properties.CustomProperties.Response;
                istraining  = data.Properties.CustomProperties.TrainingObservations;

                %Corner case 1: models that do no auto encode categoricals
                %require manual encoding with basemal.dummaryvar
                if isa(mdl, "RegressionLinear")
                    training = baseml.dummyvar( data( istraining, features ) );
                else
                    training = data( istraining, features );
                end

                %Corner case 2: Support TreeBagger using function handle
                if isa(mdl, "TreeBagger")
                    this = lime(@(x)mdl.predict( x ), training, ...
                        "Type", options.Type);
                else
                    this = lime(mdl, training);
                end

                target = data.(response)(istraining);

                if options.Type == "regression"

                    p10 = @(x)prctile( x, 10 );
                    p90 = @(x)prctile( x, 90 );

                    rng(0)
                    iD = [...
                        datasample( find( target < p10( target ) ) , 2)
                        datasample( find( target > p90( target ) ) , 2)];

                else

                    rng(0)
                    if options.Class ~= ""
                        mustBeMember(options.Class, ...
                            string(unique(data.(response))));
                        
                        iD = datasample( find( target == options.Class ) , 4 );
                    else
                        [~, iD] = datasample( target , 4 );
                    end

                end

                queryPoints = training(iD,:);

                figure("Color","W")
                tiledlayout(2,2)
                for iItem =1:4
                    iteration = this.fit( queryPoints(iItem,:), numel(features)-1, "SimpleModelType","linear");

                    nexttile()
                    f = plot( iteration );
                    f.CurrentAxes.TickLabelInterpreter = 'none';

                end %for

            end %if guard

        end %function

        function [mdl, data] = getmodel( obj, trial, mdl )
            %GETMODEL

            arguments
                obj
                trial (1,1) double {mustBeInteger,mustBePositive}
                mdl (1,1) double {mustBeInteger,mustBePositive}
            end

            if ~isempty(obj.Trials) && ~isempty(obj.Summary)

                tF_trial = ismember( obj.Trials.Trial, trial );

                %Get training data
                if any( tF_trial )
                    data = obj.Trials.Data{tF_trial};
                else
                    error("No valid trial + hour combination found.")
                end

                rowString = "Trial"+trial+"Model"+mdl;
                tF_model = ismember(obj.Summary.Properties.RowNames, rowString);

                %Get model
                if any( tF_model )


                    if isfolder(obj.ModelLocation)

                        filelist = matlab.io.datastore.DsFileSet( obj.ModelLocation, ...
                            "IncludeSubfolders", true, ...
                            "FileExtensions", ".mat").resolve.FileName;

                    else
                        error("Experiment archive not found.")
                    end

                    tF_file = contains( filelist,"Trial" + trial + "_" ) & ...
                        contains( filelist, "Model" + mdl + "_");

                    fullpathname = filelist(tF_file);

                    contents = load( fullpathname );
                    mdl = contents.result;

                else
                    error("No valid trial + model + hour combination found.")
                end

            end

        end %function

        function cm = confusionchart( obj, trial, mdl, options )
            % CONFUSIONMAT Confusion matrix for archived model
            arguments
                obj
                trial (1,1) double {mustBeInteger,mustBePositive}
                mdl (1,1) double {mustBeInteger,mustBePositive}
                options.Partition (1,1) string {mustBeMember(options.Partition, ...
                    ["", "Train", "Test"])} = "Test"
            end

            if options.Partition == "Test"
                value = 0;
            elseif options.Partition == "Train"
                value = 1;
            end
    
            if ~isempty(obj.Trials) && ~isempty(obj.Summary)
            
                [~, data] = obj.getmodel( trial, mdl );

                response = data.Properties.CustomProperties.Response;
                prediction  = ("Prediction" + mdl);

                if isprop( data.Properties.CustomProperties, "TrainingObservations" ) && options.Partition ~=""

                    tF = data.Properties.CustomProperties.TrainingObservations == value;
                    %Corner case
                    if all( tF  == 0 )
                        tF = true( height(table), 1 );
                        options.Partition = "";
                    end %if all(tF) == 0

                else
                    tF = true( height(data), 1 );
                end %if isprop

                % Calculate overall accuracy
                accuracy = (sum(data(tF,:).(response) == data(tF,:).(prediction),'omitnan') / numel(data(tF,:).(response)))*100;

                rowString = "Trial"+trial+"Model"+mdl;
                tF_model = ismember(obj.Summary.Properties.RowNames, rowString);

                modeltype = obj.Summary.modelType(tF_model);

                if options.Partition == ""
                    cmtitle = "All Data";
                else
                    cmtitle = options.Partition + " Set";
                end
                
                titlename = sprintf(cmtitle + "\n" + ...
                    "trial "+ trial + ": " + modeltype + "\n" + ...
                    "Accuracy = " + accuracy + "%%");

                figure("Color", "W")
                cm = viz.confusionchart( data(tF,:), response, prediction, ...
                    "RowSummary", 'row-normalized', ...
                    "ColumnSummary", 'column-normalized', ...
                    "Title", titlename);
            
            end %if guard

        end %function

        function predictactual(obj, trial, mdl, options)
            %PREDICTACTUAL Predict actual plot for archived model
            arguments
                obj
                trial (1,1) double {mustBeInteger,mustBePositive}
                mdl (1,1) double {mustBeInteger,mustBePositive}
                options.Partition (1,1) string {mustBeMember(options.Partition, ...
                    ["", "Train", "Test"])} = "Test"
            end

            if options.Partition == "Test"
                value = 0;
            elseif options.Partition == "Train"
                value = 1;
            end
    
            if ~isempty(obj.Trials) && ~isempty(obj.Summary)
            
                [~, data] = obj.getmodel( trial, mdl );

                response = data.Properties.CustomProperties.Response;
                prediction  = ("Prediction" + mdl);

                if isprop( data.Properties.CustomProperties, "TrainingObservations" ) && options.Partition ~=""

                    tF = data.Properties.CustomProperties.TrainingObservations == value;
                    %Corner case
                    if all( tF  == 0 )
                        tF = true( height(table), 1 );
                        options.Partition = "";
                    end %if all(tF) == 0

                else
                    tF = true( height(data), 1 );
                end %if isprop

                % Calculate overall accuracy
                error = (data(tF,:).(prediction) - data(tF,:).(response));
                rmse = sqrt(mean(error.^2));

                rowString = "Trial"+trial+"Model"+mdl;
                tF_model = ismember(obj.Summary.Properties.RowNames, rowString);

                modeltype = obj.Summary.modelType(tF_model);

                if options.Partition == ""
                    cmtitle = "All Data";
                else
                    cmtitle = options.Partition + " Set";
                end
                
                titlename = sprintf(cmtitle + "\n" + ...
                    "trial "+ trial + ": " + modeltype + "\n" + ...
                    "RMSE = " + rmse);

                figure("Color", "W")
                viz.predictactual(data(tF, :), response, prediction, ...
                    "Title", titlename);
            
            end %if guard

        end
    
    end %methods

    methods ( Access = private )

        function value = i_customRead( ~, file )

            contents = load( file );

            parts       = strsplit(fileparts(file),filesep);
            dsName      = string( parts(end) );
            trialName   = double( extractBetween(string(file), "Trial", "_") );

            value = table();
            value.Experiment = dsName;
            value.Trial      = trialName;
            value.Data       = { contents.data };

        end %function

    end %methods

end %classdef
