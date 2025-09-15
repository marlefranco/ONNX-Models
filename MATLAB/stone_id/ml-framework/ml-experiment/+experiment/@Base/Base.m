classdef (Abstract, Hidden) Base < matlab.mixin.SetGet
    %experiment.Base Abstract base class for experiment package
    %
    
    %   N.C.Howes, Sudheer Nuggehalli
    %   Copyright 2021 The MathWorks Inc.
    %
    
    
    properties
        %Data Input dataset as a table. Input data will prepared using the
        %specified data preparation pipeline.
        Data table
        %DataFcn Specify data preparation pipeline as anonymous function
        %@(x,settings)somePipeline(x,settings{:}). Note that the pipeline
        %function must be on path.
        DataFcn (1,1) function_handle {mustBeClass(DataFcn,"function_handle")} = @(x)x;
        %DataConfiguration Configure 'sweepable' parameters for data
        %preparation pipeline as vector of optimizeParameter objects.
        %Default is empty.
        DataConfiguration (1,:) optimizeParameter
        %ModelConfiguration Configure 'sweepable' parameters for
        %selected model as vector of optimizeParameter objects. Default is empty.
        ModelConfiguration (1,:) optimizeParameter
        %AdditionalConfiguration Configure data preparation pipelines with
        %parameters that will not be used in the experiment design as
        %vector of constantParameter objects. Default is empty.
        AdditionalConfiguration (1,:) constantParameter
        %Description String/tag associated with the experiment session
        Description (1,:) string
        %Name of experiment session
        Name (1,1) string
        %Search Specify parameter selection method for configuration. Valid
        %options are: "gridsearch" and "randomsearch".
        Search (1,1) string {mustBeMember(Search, ["gridsearch", "randomsearch"])} = "gridsearch"
        %Evaluations Specify number of samples as integer value if "Search"
        %is "randomsearch" (default = 30).
        Evaluations (1,1) double {mustBeInteger} = 30;
        %NumGridDivisions Specify number of grid samples per dimension if
        %"Search" is "gridsearch" (default = 10).
        NumGridDivisions (1,1) double {mustBeInteger} = 2;
        %CustomProperties Specify any custom properties in the data pipeline to
        %appear in evaluation metrics. Note that these custom metrics MUST
        %be defined in the CustomProperties of the table that is output by
        %the data preparation pipeline.
        CustomProperties (1,:) string = ""
        %UseParallel Flag for setting parallel preferences
        UseParallel (1,1) logical = false
        
    end
    
    properties (SetAccess = protected, GetAccess = public, Hidden )
        Trials experiment.item.Base
    end
    
    properties (SetAccess = private, GetAccess = public, Hidden )
        Samples
        Constants
    end
    
    properties ( Dependent, Hidden )
        DataParameters
        ModelParameters
        AdditionalParameters
        nSamples
        validTrialIndices
    end
    
    methods (Abstract)
        fit(obj)
    end
    
    methods
        function obj = Base( varargin )
            %(Abstract)superclass constructor
            
            if nargin > 0
                
                if ~isempty( varargin )
                    set(obj, varargin{:})
                end
                
            end
            
        end
        
        function delete(obj)
            %Destructor
            
            delete(obj)
            
        end
        
    end %constructor
    
    methods
        function configure( obj )
            %CONFIGURE Validate and buid experiment
            
            obj.validate()
            obj.build()
            
        end %function
        
        function  build( obj )
            %BUILD Summary of this method goes here
            
            %Generate configurations
            if ~isempty( obj.DataConfiguration ) || ~isempty( obj.ModelConfiguration )
                obj.sample();
            end
            
            %Assign constants
            if ~isempty( obj.AdditionalConfiguration )
                obj.i_buildconstants();
            end
            
            %Create experiment items
            trials = experiment.item.Supervised.empty(0,1);
            for i = 1:obj.nSamples
                trials(i) = experiment.item.Supervised();
                set(trials(i), 'Data', obj.Data);
            end
            set(trials, {'pipeline', 'model'}, {obj.DataFcn, obj.Model});
            set(trials, {'pipelinesettings'}, obj.i_pipelinesettings)
            set(trials, {'modelsettings'}, obj.i_modelsettings)
            
            %Assign the trials
            obj.Trials = trials;
            
        end %function
        
        
        function reset( obj )
            %RESET TODO DOC
            
            for i = 1:numel( obj.Trials )
                obj.Trials(i).reset();
                obj.Trials(i).Data = obj.Data;
                obj.Trials(i).Prepared = false;
            end
            
        end %reset
        
        
        function run(obj)
            %RUN Run experimental items/configurations. Each item run
            %consists of a unique data preparation pipeline and model
            %configuration.
            %
            % Syntax:
            %
            % obj.run() run all experiment configurations. This method is
            % equivalent to running obj.prepare() + obj.fit()
            %
            
            % Build data pipelines and train ML models
            obj.build()
            obj.prepare();
            obj.fit();
            
        end %function
        
        
        function prepare(obj)
            %PREPARE Run data preparation pipelines for items/configuration
            %in the experiment.
            %
            % Syntax:
            %
            %   obj.prepare() run all data pipelines.
            %
            %
            %
            
            %Validate data pipeline configuration
            obj.validateData();
            
            if ~isempty( obj.Trials )
                
                obj.reset()
                
                try
                    for i = 1:numel( obj.Trials )
                        
                        trial = obj.Trials(i);
                        
                        pipeFunction = trial.pipeline;
                        pipeSettings = trial.pipelinesettings;
                        
                        % Note: Think about managing pipesettings to
                        % provide more flexibility.
                        trial.Data = pipeFunction( obj.Data, pipeSettings );
                        trial.Prepared = true;
                        
                    end %for iTrial
                    
                catch ME
                    error(ME.identifier,'%s', ME.message)
                end
                
                %Run post preparation checks
                obj.i_postprepareChecks();
                
            end
            
        end %function
        
        
        
        function value = preview( obj, options )
            %PREVIEW Preview the experimental items/configuration.
            %
            % Syntax:
            %
            % value = obj.preview() preview all items
            %
            % value = obj.preview("PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %
            %   "Id" - preview specified items. Id is specified as an
            %   integer scalar or array and corresponds to TrialID in
            %   the RowName of obj.preview().
            %
            
            arguments
                obj experiment.Base
                options.Id (1,:) double {mustBeInteger} = 1:obj.nSamples;
            end
            
            
            %Preview Preview experiment design/runs
            if ~isempty( obj.Samples )
                value = struct2table( obj.Samples( options.Id ), 'AsArray', true );
                value.Properties.RowNames = "Trial" + (options.Id);
            else
                value = table();
            end
            
        end %function
        
        function value = fullpreview( obj )
            
            arguments
                obj experiment.Base
            end
            
            if ~isempty( obj.Trials ) && ~isempty( obj.validTrialIndices )
                
                trials = obj.Trials;
                
                if isprop(trials(1).Data.Properties.CustomProperties, "Options")
                    
                    values = cell( numel(trials), 1);
                    for iTrial = 1: numel(trials)
                        
                        this = struct2table(...
                            trials(iTrial).Data.Properties.CustomProperties.Options, ...
                            "AsArray", true);
                        
                        % Corner case
                        tF = find( varfun(@(x)istable(x) &~iscell(x), this,"OutputFormat", "uni") );
                        if any(tF)
                            vars = string(this.Properties.VariableNames(tF));
                            for iVar = vars(:)'
                                this.(iVar) = {this.(iVar)};
                            end
                        end
                        
                        if iTrial > 1
                            names = intersect(values{iTrial-1}.Properties.VariableNames, ...
                                this.Properties.VariableNames,'stable');
                            
                            values{iTrial} = this(:,names);
                            
                        else
                            
                            values{iTrial} = this;
                            
                        end
                        
                    end
                    
                    value = vertcat(values{:});
                    
                    value.Properties.RowNames = "Trial" + (obj.validTrialIndices);
                    
                    other= obj.preview( "Id", obj.validTrialIndices);
                    if ~isempty(other)
                        names =  setdiff( other.Properties.VariableNames, ...
                            value.Properties.VariableNames, 'stable');
                        
                        value = [value other(:,names)];
                    end
                    
                else
                    
                    value = table();
                    
                end %if isprop
                
            else
                
                value = table();
                
            end %if
            
        end %function
        
        
        function value = view(obj, options )
            %VIEW View data in an experimental item/configuration. View
            %will return the current state of the data. If view is
            %called prior to running the pipeline or models, it will return
            %the original data. If called after running the data
            %pipelines and/or modelf fits it will return the result.
            %
            % Syntax:
            %
            % value = obj.view() view current state of data an experiment
            % item (default = 1).
            %
            % value = obj.view("PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %
            %   "Id" - View data in specified item. Id is specified as an
            %   integer scalar and corresponds to TrialID in
            %   the RowName of obj.preview().
            %
            
            arguments
                obj
                options.Id (1,1) double {mustBeInteger} = 1
            end
            
            value = table();
            
            if ~isempty( obj.Trials )
                value = obj.Trials( options.Id ).Data;
            end
            
        end %view
        
        
        function value = head( obj, options )
            %HEAD TODO
            
            arguments
                obj
                options.Id (1,1) double {mustBeInteger} = 1
            end
            
            value = table();
            if ~isempty( obj.Trials )
                value = head( obj.Trials( options.Id ).Data );
            end
            
        end %function
        
        
        function [value, trials, models] = sort( obj, options )
            %SORT Rank experiment ML model results by an evaluation metric
            %
            % Syntax:
            %
            % value = obj.sort() rank model results on cross-validated mean
            % square error (mseOnCV).
            %
            % [value, runId, modelId] = obj.sort() return a ranked table
            % and corresponding ID for the experiment trial and model
            % result.
            %
            % [value, ...] = obj.sort("PARAM1", value1, ... ) specifies
            % optional parameter name/value pairs:
            %
            %   "Metric" - evaluation metric used to rank model results.
            %   Valid options are: mseOnCV, mseOnTest, or mseOnResub
            %
            
            arguments
                obj
                options.Metric
            end
            
            results = obj.describe( );
            
            if ~isempty( obj.Trials ) && ~isempty( results )
                
                if matches(obj.Type, ["Classification", "SemiSupervised"])
                    if ~contains(options.Metric, "error")
                        metricTable = results.(options.Metric);
                        [~, index] = sortrows(metricTable, "Avg", "descend");

                        value = results(index, :);
                    else
                        value = sortrows( results, options.Metric );
                    end
                else
                    value = sortrows( results, options.Metric );
                end
                
                [trials, models] = obj.ind2sub( value.Item );
                
            else
                value = table();
                trials = [];
                models = [];
            end %if ~isempty
            
        end %sort
        
        
        function value = describe( obj, detail )
            %DESCRIBE Compare input parameters with model evaluation
            %critera
            %
            % Syntax:
            %
            % value = obj.describe() input parameters sweep with evaluation results for
            % all experiment runs
            %
            % value = obj.describe( "full" ) all input parameters with evaluation results for
            % all experiment runs
            %
            
            arguments
                obj
                detail (1,1) string {mustBeMember(detail,["sweep", "full"])} = "sweep";
            end
            
            value = table();
            
            selection = obj.validTrialIndices;
            
            if ~isempty( obj.Trials ) && ~isempty( selection )
                
                nModels = [];
                for iItem = 1:numel(selection)
                    nModels = [nModels numel( obj.Trials( selection(iItem) ).items )]; %#ok<AGROW>
                end
                
                value = table();
                if any( nModels ~= 0 )
                    
                    if detail == "full"
                        value = varfun(@(x)repelem(x,nModels,1), obj.fullpreview() );
                    else
                        value = varfun(@(x)repelem(x,nModels,1), obj.preview( "Id", selection) );
                    end
                    
                    value.Properties.VariableNames = ...
                        strrep( value.Properties.VariableNames, "Fun_", "" );
                    
                    metrics = obj.Trials(selection).describe();
                    tF = ismember(metrics.Properties.VariableNames, "modelType");
                    
                    %Add an item id
                    Item = 1:height( metrics(:,tF)); Item = Item(:);
                    items = table(Item);
                    
                    value = [ items metrics(:,tF) value metrics(:,~tF)];
                    
                    modelId = cellfun(@(x)1:x, num2cell(nModels),'uni',0);
                    modelId = cat(2,modelId{:});
                    
                    value.Properties.RowNames = "Trial"+repelem(selection(:), nModels,1)...
                        +"Model"+modelId(:);
                    
                end
                
                
                %Append additional custom properties
                if any( obj.CustomProperties ~= "" )
                    
                    customs = obj.Trials.custom( obj.CustomProperties, "Id", selection );
                    
                    if nModels > 0
                        customs = varfun(@(x)repelem(x,nModels,1), customs);
                        customs.Properties.VariableNames = ...
                            strrep( customs.Properties.VariableNames, "Fun_", "" );
                    end
                    
                    value = [value customs];
                    
                end
                
            end %if
            
        end %describe
        
        
        function [value, info] = select(obj, item, options)
            % SELECT Select/extract a model from the experiment
            %
            % Syntax:
            %
            % value = obj.select() return a model from the results. Default
            % is the first model from the first experiment configration as
            % listed in obj.describe.
            %
            % [value, info] = obj.select() returns a model and the
            % evaluation criteria as a table.
            %
            % [value, ...] = obj.select(item, "PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %
            %   "item"      - index to a specific experiment item
            %   corresponding see obj.describe() to list available items.
            %
            %   %Metadata - (default false). If true, returns a structure
            %   containing the model, model name, and
            %   configuration/options.
            %
            
            arguments
                obj experiment.Base
                item (1,1) double {mustBeInteger} = 1
                options.Metadata (1,1) logical = false
            end
            
            if ~isempty( obj.validTrialIndices )
                
                [ trials, models ] = obj.ind2sub( item );
                options.Id = trials;
                options.Model = models;
                
                args = namedargs2cell( options );
                [value, info] = obj.Trials.select( args{:} );
                
            else
                value = [];
                info  = [];
            end
            
        end %select
        
        
        %         function value = Pipelines( obj )
        %             %PIPELINES TODO DOC
        %
        %             value = [obj.Trials.result];
        %
        %         end %pipelines
        
        function validate( obj )
            %VALIDATE Validate data pipeline and model configuration
            %
            %   Syntax:
            %
            %   obj.validate() validate experiment configuration parameters
            %
            %
            
            %Validate Data Configuration
            obj.validateData();
            
            %Validate Model Parameters
            obj.validateModel();
        end
        
        function validateData( obj )
            %VALIDATE Validate data pipeline ad configuration
            %
            %   Syntax:
            %
            %   obj.validateData() validate experiment configuration parameters
            %
            %
            
            if ~isdeployed
                % Data preparation parameters
                parameters = [ ...
                    [obj.DataConfiguration.Name] ...
                    [obj.AdditionalConfiguration.Name] ...
                    ];
                
                %Validate data preparation parameters
                if ~isempty( parameters )
                    
                    %Parse data preparation function
                    str  = func2str( obj.DataFcn );
                    expr = "@*\)([a-zA-Z]{1,1}[a-zA-Z_0-9.]+)(*";
                    
                    filename = string(...
                        regexp(str, expr, "tokens")...
                        );
                    
                    %Check if function found on path
                    if isempty( which(filename) )
                        error( 'Data preparation function: %s is not found on path.', filename )
                    end
                    
                    validparameters = validatePipelineParameters( filename );
                    
                    parameters = [ ...
                        [obj.DataConfiguration.Name] ...
                        [obj.AdditionalConfiguration.Name] ...
                        ];
                    
                    isnotvalid = ~ismember( parameters, validparameters );
                    
                    if any( isnotvalid )
                        error( ['Unhandled/unknown parameters in configuration:', ...
                            sprintf( '\n''%s''', parameters(isnotvalid))] )
                    end % if any(
                    
                end %if ~isempty( parameters )
            end
            
        end
        
        function validateModel(obj)
            %VALIDATE Validate model configuration
            %
            %   Syntax:
            %
            %   obj.validateModel() validate experiment models
            %
            %
            
            %Model preparation parameters
            modelparameters = [obj.ModelConfiguration.Name];
            
            %Validate auto ml parameters
            if ~isempty( modelparameters )
                
                %Special case: Validate model parameters if automl method
                switch obj.Model
                    case {"automl","selectml","stackml"}
                        
                        validparameters = obj.ValidModelParameters;
                        
                        tF = all( ismember(modelparameters, validparameters) );
                        
                        if ~tF
                            
                            list = setdiff(modelparameters, validparameters);
                            
                            error( ['Invalid ModelConfiguration parameters: %s.\n'...
                                'Please select: %s\n'], join(list, ", "), join(validparameters,", ") )
                        end
                        
                    otherwise
                        %TODO Add validation for model types
                        
                end %switch
                
            end %if ~isempty( modelparameters )
            
        end %function
        
        
        function [statusok, filename] = save( obj, item, options, paths )
            %SAVE Save experiment to disk
            %
            % Syntax:
            %
            % statusok = obj.save() save experiment items / configurations
            % to disk as a *.mat file. File contains a table with input
            % parameters, pipeline result, and model results for each
            % experiment item. File also contains three convenience
            % functions to access model results, experiment metadata, and
            % item/configuration metadata.
            %
            % statusok = obj.save( "PARAM1", value1, ... ) specifies
            % optional parameter name/value pairs:
            %
            %   "Id" - index specific experiment items. Id is specified
            %   as integer scalar or array and corresponds to TrialID in
            %   the RowName.
            %
            %   "Model"   - index to a specific model result. Model is
            %   specified as an integer scalar and corresponds to the
            %   ModelID in the RowName.
            %
            %   "WriteDirectory" - export directory
            %
            %   "FileName" - filename
            %
            %   %Metadata - (default false). If true, returns a structure
            %   containing the model, model name , data configuration, model
            %   configuration, and a data table containing the result of
            %   the data preparation steps and fit predictions.
            %
            
            arguments
                obj
                item (1,:) {mustBeIntegerOrAll}
                options.Metadata (1,1) logical = false
                paths.WriteDirectory (1,1) string = fullfile( tempdir, "mlexperiment" )
                paths.FileName (1,1) string = ""
            end
            
            statusok = false;
            
            args = namedargs2cell( options );
            
            if isstring( item )
                item = 1:height( obj.describe );
            end
            
            if ~isfolder( paths.WriteDirectory )
                mkdir( paths.WriteDirectory )
            end
            
            if ~isempty( obj.validTrialIndices )
                
                for iItem = item(:)'
                    
                    [ trialId, modelId ] = obj.ind2sub( iItem );
                    
                    [result, info] = obj.select( iItem, args{:} );
                    
                    if ~isempty( result )
                        
                        if paths.FileName == ""
                            filename = "ExperimentML_" + "Trial" + trialId + "_" ...
                                + "Model" + modelId + "_" ...
                                + string( datetime('now', 'format', 'MMMddyyyy') ) + ".mat";
                        else
                            filename = paths.FileName + "_" + "Trial" + trialId + "_" ...
                                + "Model" + modelId + "_" ...
                                + string( datetime('now', 'format', 'MMMddyyyy') ) + ".mat";
                        end %if paths
                        
                        save( fullfile( paths.WriteDirectory, filename ), ...
                            "result", "info", ...
                            "-v7.3");
                        
                        statusok = true;
                        
                    end %if ~isempty
                    
                end %for iItem
                
                if options.Metadata
                    if paths.FileName == ""
                        filename = "ExperimentML_" + "Metadata_" +  ...
                            string( datetime('now', 'format', 'MMMddyyyy') ) + ".xlsx";
                    else
                        filename = paths.FileName + "Metadata_" + ...
                            string( datetime('now', 'format', 'MMMddyyyy') ) + ".xlsx";
                    end %if paths

                    this = obj.describe("full");
                    writetable(splitvars(this), fullfile( paths.WriteDirectory, filename), "WriteRowNames", true );
                end
                
            end %if ~isempty( obj.validTrialIndices )
            
        end %function
        
        
        function statusok = writedata( obj, trial, paths )
            %WRITEDATA
            
            arguments
                obj
                trial (1,:) {mustBeIntegerOrAll}
                paths.WriteDirectory (1,1) string = fullfile( tempdir, "mlexperiment" )
                paths.FileName (1,1) string = ""
            end
            
            statusok = false;
            
            if isstring( trial )
                trial = 1:numel( obj.Trials );
            end
            
            if ~isfolder( paths.WriteDirectory )
                mkdir( paths.WriteDirectory )
            end
            
            if ~isempty( obj.validTrialIndices )
                
                for iTrial = trial(:)'
                    
                    if paths.FileName == ""
                        filename = "ExperimentML_" + "Trial" + iTrial + "_" ...
                            + string( datetime('now', 'format', 'MMMddyyyy') ) + ".mat";
                    else
                        filename = paths.FileName + "_" + "Trial" + iTrial + "_" ...
                            + string( datetime('now', 'format', 'MMMddyyyy') ) + ".mat";
                    end %if paths
                    
                    data = obj.view("Id", iTrial);
                    
                    save( fullfile( paths.WriteDirectory, filename ), ...
                        "data", "-v7.3");
                    
                end %for iTrial
                
                statusok = true;
                
            end %if ~isempty
            
        end %function
        
        
        function plotmetric( obj, options )
            %PLOTMETRIC Plot evulation metric for all experimental runs as a
            %bar graph.
            %
            % Syntax:
            %
            % obj.plotmetric()
            %
            % obj.plotmetric("PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %   "Metric" - Evaluation critera used in visualization.
            %              Specify as a scalar string. Valid options
            %              are:   "r2OnTrain","rmseOnTrain", ...
            %                     "errorOnResub","errorOnCV", "errorOnTest", ...
            %                     "f1ScoreOnTrain","f1ScoreOnTest", ...
            %                     "AUCOnTrain","AUCOnTest", ...
            %                     "mseOnCV", "mseOnResub" "mseOnTest".
            %
            %   "Sort"  - Sort experimental runs by evaluation criteria.
            %             Specify as a logical (default = true).
            %
            
            arguments
                obj
                options.Metric (1,1) string
                options.Sort
                options.Axes (1,1) {mustBeA(options.Axes, ...
                    ["matlab.graphics.axis.Axes","matlab.ui.control.UIAxes"])} = gca()
            end
            
            modelmetrics = obj.describe();
            
            if ~isempty(modelmetrics)
                
                if contains(options.Metric, ["f1", "AUC"])
                    metric = modelmetrics.(options.Metric).Avg;
                else
                    metric =   modelmetrics.(options.Metric);
                end
                
                titleStr = options.Metric;
                
                models =  extractBetween( modelmetrics.modelType,'(',')');
                
                tF = contains(modelmetrics.modelType, "Stack" );
                if any( tF )
                    models( tF ) = models(tF) + "stack";
                end
                
                trials = extractBefore( modelmetrics.Properties.RowNames, "Model");
                labels = trials + " " + models;
                labels = strrep( labels, "Trial", "trial" );
                
                if options.Sort == true
                    
                    [~, index] = sort( metric );
                    
                    labels = labels(index);
                    metric = metric(index);
                end
                
                %                 ax = gca();
                ax = options.Axes;
                bar(ax, metric )
                
                ylabel(ax, options.Metric )
                ax.XTick = 1:numel(metric);
                ax.XTickLabel = labels;
                ax.XTickLabelRotation = 45;
                title(ax, titleStr )
            end
            
        end %plotmetric
        
    end %public methods
    
    
    methods (Hidden)
        
        function resetmodels( obj )
            %RESETMODELS TODO DOC
            
            for i = 1:numel( obj.Trials )
                trial = obj.Trials(i);
                
                if ~isempty( trial.model )
                    
                    trial.model = [];
                    
                    tf = contains(trial.Data.Properties.VariableNames, "Prediction");
                    trial.Data( :, tf ) = [];
                    
                end
                
            end
            
        end %function
        
    end %hidden methods
    
    
    methods
        function value = get.nSamples( obj )
            
            if isempty( obj.Samples )
                value = 1;
            else
                value = numel( obj.Samples );
            end
            
        end %get.nSamples
        
        function value = get.validTrialIndices( obj )
            
            if ~isempty( obj )
                value = find( [obj.Trials.Prepared] == true );
            else
                value = [];
            end
        end %get.validTrialIndices
        
        function value = get.DataParameters( obj )
            if ~isempty( obj.DataConfiguration )
                value = [obj.DataConfiguration.Name];
            else
                value = "";
            end
        end %get.DataParameters
        
        function value = get.ModelParameters( obj )
            if ~isempty( obj.ModelConfiguration )
                value = [obj.ModelConfiguration.Name];
            else
                value = "";
            end
        end %get.ModelParameters
        
        function value = get.AdditionalParameters( obj )
            if ~isempty( obj.ModelConfiguration )
                value = [obj.AdditionalConfiguration.Name];
            else
                value = "";
            end
        end %get.AdditionalParameters
        
    end %get methods
    
    
    methods ( Access = protected )
        
        function sample( obj )
            %SAMPLE Internal function performs gridsearch or randomsearch
            %over optimizable data and model parameters
            
            parameters = [obj.DataConfiguration(:)' obj.ModelConfiguration(:)'];
            
            results = cell( size( parameters ) );
            levels = cell( size( parameters ) );
            
            types = [parameters.Type];
            tF = ismember(types, ["optimizeContinuous" "optimizeInteger"]);
            
            switch obj.Search
                
                case "gridsearch"
                    
                    %Continuous
                    [parameters(tF).Method] = deal("grid");
                    [parameters(tF).Samples] = deal(obj.NumGridDivisions);
                    
                    %Discrete
                    [parameters(~tF).Method] = deal("unique");
                    
                case "randomsearch"
                    
                    %Continuous
                    [parameters(tF).Method] = deal("rand");
                    [parameters(tF).Samples] = deal(obj.Evaluations);
                    
                    %Discrete
                    [parameters(~tF).Method] = deal("rand");
                    [parameters(~tF).Samples] = deal( obj.Evaluations );
                    
            end %switch
            
            for iObj = 1 : numel( parameters )
                
                param = parameters( iObj );
                
                results{iObj} = param.sample();
                levels{iObj}  = 1 : numel( results{iObj} );
                
            end
            
            result = struct();
            switch obj.Search
                case "gridsearch"
                    
                    if license( "test", "neural_network_toolbox" )
                        sampleArray = combvec( levels{:} );
                    else
                        c = cell(1, numel( levels ) );
                        [c{:}] = ndgrid( levels{:} );
                        trials = cell2mat( cellfun(@(v)v(:), c, 'UniformOutput',false) );
                        sampleArray = transpose( trials );
                    end
                    
                    nSample = size(sampleArray, 2);
                    
                    for iObj = 1 : numel( parameters )
                        
                        name = parameters(iObj).Name;
                        
                        if iscell( results{iObj}( sampleArray(iObj,:) ) )
                            values = results{iObj}( sampleArray(iObj,:) );
                        else
                            values = num2cell( results{iObj}( sampleArray(iObj,:) ) );
                        end  %if iscell
                        
                        [result(1:nSample).(name)] = deal( values{:} );
                        
                    end
                    
                case "randomsearch"
                    
                    nSample = obj.Evaluations;
                    
                    for iObj = 1 : numel( parameters )
                        name = parameters(iObj).Name;
                        
                        if iscell( results{iObj} )
                            values = results{iObj};
                        else
                            values = num2cell( results{iObj} );
                        end
                        
                        [result(1:nSample).(name)] = deal( values{:} );
                    end
                    
            end
            
            obj.Samples = result;
            
        end %sample
        
        
        function i_buildconstants( obj )
            %I_BUILDCONSTANTS Internal function
            
            constants = struct();
            nSample = obj.nSamples;
            
            parameters = obj.AdditionalConfiguration;
            for iObj = 1 : numel( parameters )
                name = parameters(iObj).Name;
                
                if iscell( parameters(iObj).Value )
                    values = repmat( parameters(iObj).Value, 1, nSample );
                else
                    values = repmat({parameters(iObj).Value}, 1, nSample );
                end
                
                [constants(1:nSample).(name)] = deal( values{:} );
            end
            
            obj.Constants = constants;
            
        end %i_buildconstants
        
        
        function i_postprepareChecks( obj )
            
            for  i = 1:numel( obj.Trials )
                
                trial = obj.Trials(i);
                
                %Validate standard custom properties definition in preparation
                %pipeline
                tf_Feature  = isprop( trial.Data.Properties.CustomProperties, "Features" );
                tf_Response = isprop( trial.Data.Properties.CustomProperties, "Response" );
                tf_TrainObs = isprop( trial.Data.Properties.CustomProperties, "TrainingObservations" );
                tF_Options  = isprop( trial.Data.Properties.CustomProperties, "Options" );
                
                if ~tf_Feature
                    warning( ['List of features is undefined in data preparation pipeline. ' ...
                        'If undefined, all data variables are mapped/considered predictor variables except the last.'...
                        'Please see newPipelineTemplate for example of defining feature list.'])
                end %if tF_Feature
                
                if ~tf_Response
                    warning( ['Response variable is undefined in data preparation pipeline. ' ...
                        'If undefined, the last variable in the table is considered the response.'...
                        'Please see newPipelineTemplate for example of defining the response.'])
                end %if tF_Response
                
                if ~tf_TrainObs
                    warning( ['Train/Test partition is undefined. ' ...
                        'If undefined, all observations are used in training and no test set is evaluated.'...
                        'Please see newPipelineTemplate for example of defining a data partition.'])
                end %if tF_TrainObs
                
                if ~tF_Options
                    warning( ['Pipeline options are not stored in custom properties. ' ...
                        'If undefined, you will not have a record of the options used to configure experiment trial.'...
                        'Please see newPipelineTemplate for example of storing pipeline options.'])
                end %if tF_Options
                
                %Validate data
                if tf_Feature
                    features = trial.Data.Properties.CustomProperties.Features;
                else
                    features = string( trial.Data.Properties.VariableNames(1:end-1) );
                end %if tf_Feature
                
                % convert features to row vector if needed
                if ~isrow(features)
                    features = transpose(features);
                end
                
                if tf_Response
                    response = trial.Data.Properties.CustomProperties.Response;
                else
                    response = string( trial.Data.Properties.VariableNames(end) );
                end %if tf_Response
                
                if ~isrow(features)
                    features = transpose(features);
                end
                
                if obj.Type == "Classification"
                    datasummary = util.summarize( trial.Data(:,[features response]), "Response", response );
                else
                    datasummary = util.summarize( trial.Data(:,[features response]));
                end %if obj.Type
                
                if datasummary.tF_Missing
                    warning( "Training data contain NaN." )
                end
                
                if datasummary.tF_Constant
                    warning( "Training data contain constant variables." )
                end
                
                if datasummary.tF_ConstantInClass == true
                    warning( "Training data contain constant variables in one or more response classes." )
                end
                
                if datasummary.tF_CategoricalNotConvert
                    warning( ['Training data contain discrete variables represented as cellstr, strings, or chars.'...
                        'Please convert to categorical for consistent behavior between training and prediction.'] )
                end
                
            end %for iTrial
            
        end %function
        
    end %protected
    
    
    methods ( Access = protected )
        function settings = i_pipelinesettings(obj)
            
            %Get samples
            samples = obj.preview();
            
            %Prepare data parameters
            if ~isempty( obj.DataConfiguration )
                parameters = ismember( samples.Properties.VariableNames, ...
                    [obj.DataConfiguration.Name] );
            else
                parameters = table();
            end
            
            settings = cell(obj.nSamples,1);
            for iSample = 1 : obj.nSamples
                
                if ~isempty( obj.DataConfiguration )
                    value = namedargs2cell( table2struct( samples(iSample,parameters) ) );
                else
                    value = {};
                end
                
                if ~isempty( obj.AdditionalConfiguration )
                    constants = namedargs2cell(obj.Constants( iSample ) );
                    value = [value  constants];  %#ok<AGROW>
                end
                
                settings{iSample} = value;
                
                
            end %for iSample
            
            
        end %function
        
        
        function settings = i_modelsettings(obj)
            
            %Get samples
            samples = obj.preview();
            
            if ~isempty( obj.ModelConfiguration )
                parameters = ismember( samples.Properties.VariableNames, ...
                    [obj.ModelConfiguration.Name] );
            else
                parameters = table();
            end
            
            settings = cell(obj.nSamples,1);
            for iSample = 1 : obj.nSamples
                
                if ~isempty( obj.ModelConfiguration )
                    value = namedargs2cell( table2struct(samples(iSample,parameters)) );
                else
                    value = {};
                end
                
                settings{iSample} = value;
                
            end %for iSample
            
        end %function
        
        
        function [I,J] = ind2sub( obj, iD )
            
            if ~isempty( obj.validTrialIndices )
                
                table = obj.describe();
                value = table( iD,: );
                
                %Special case
                if obj.Type == "Unsupervised"
                    I = double( string( extractBetween( value.Properties.RowNames, "Trial", "Label" ) ) );
                    J = double( string( extractAfter( value.Properties.RowNames, "Label" ) ) );
                else
                    I = double( string( extractBetween( value.Properties.RowNames, "Trial", "Model" ) ) );
                    J = double( string( extractAfter( value.Properties.RowNames, "Model" ) ) );
                end % obj.Type
                
            end %if ~isempty
            
        end %function
        
    end %methods
    
end %experiment.Base

%Local Validation functions
function mustBeClass(a,b)
if class(a) ~= b
    error('Invalid class')
end
end %function

function mustBeIntegerOrAll( value )

if isstring(value)
    if value ~= "all"
        error("Please specify numeric integers or ""all""")
    end
else
    mustBeInteger(value)
end

end %function
