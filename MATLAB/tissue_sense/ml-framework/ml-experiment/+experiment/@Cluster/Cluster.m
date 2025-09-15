classdef Cluster < experiment.Base & experiment.mixin.Cluster
    %experiment.Cluster Create a clustering experiment.
    %
    %   experiment.Cluster methods:
    %
    %   validate      - validate the input configruation (DataConfiguration)
    %   build         - construct experimental runs from input configuration
    %   preview       - preview the experimental runs as a table
    %
    %   prepare       - run the data preparation pipelines
    %   fit           - fit model to data pipeline result
    %   run           - prepare and train in a single step
    %
    %   describe      - view input parameters with model results as a table
    %   sort          - rank runs by user defined metric
    %   select        - select/extract a model from the experiment
    %   view          - view dataset corresponding to an experiment run
    %
    %   ploterror     - visualize model error as bar graph
    %
    %   save          - export experiment to disk
    %
    %   N.C. Howes
    %   MathWorks Consulting 2020
    %
    %
    
    properties
        Model (1,1) string ...
            {mustBeMember(Model,[...
            "automl"
            "dbscan"
            "gmm"
            "hierarchical"
            "kmeans"
            "kmedoids"
            "som"
            "spectral"
            ])} = "som"
    end
    
    properties (Constant)
        Type = "Unsupervised"
    end
    
    properties ( SetAccess = private, Hidden )
        ValidModelParameters =  "Learners"
    end
    
    methods
        function obj = Cluster(varargin)
            %experiment.Cluster constructor
            
            obj = obj@experiment.Base( varargin{:} );
            
        end %experiment.Cluster
    end %constructor
    
    methods        
        function fit(obj)
            %FIT Train specified ml for items in items/configuration in the
            %experiment. Note the data preparation via obj.prepare() must occur
            %prior to fit.
            %
            % Syntax:
            %
            % obj.fit() train specified ml for all items in the experiment.
            %
            
            %Validate Model Parameters 
            obj.validateModel();
            
            if ~isempty( obj.Trials )
                obj.resetmodels()
                model = obj.Model;
                obj.( model );
            end %if ~isempty
            
        end %function

        
        function value = describe(obj, detail)
            %DESCRIBE Compare input parameters with model evaluation
            %critera
            %
            % Syntax:
            %
            % value = obj.describe() input parameters and evaluation results for
            % all experiment runs
            %
            % value = obj.describe( "full" ) all input parameters with evaluation results for
            % all experiment runs
            %
            
            arguments
                obj
                detail (1,1) string {mustBeMember(detail,["sweep", "full"])} = "sweep";
            end
            
            value = describe@experiment.Base( obj, detail );
            value.Properties.RowNames = strrep(value.Properties.RowNames, "Model", "Label"); 
            
        end %function
        
        
        function value = sort( obj, options )
            %SORT Rank experiment ML model results by an evaluation metric
            %
            % Syntax:
            %
            % value = obj.sort() rank model results on TODO
            %
            % [value, runId, modelId] = obj.sort() return a ranked table
            % and corresponding ID for the experiment item and model
            % result.
            %
            % [value, ...] = obj.sort("PARAM1", value1, ... ) specifies
            % optional parameter name/value pairs:
            %
            %   "Metric" - evaluation metric used to rank model results.
            %   Valid options are: TODO.
            %
            
            arguments
                obj
                options.Metric (1,1) string {mustBeMember(options.Metric, ...
                    "avg_Score")} = "avg_Score"
            end
            
            args  = namedargs2cell( options );
            
            value = sort@experiment.Base( obj, args{:} );
            
            value = flipud(value);
            
        end %function
        
        
        function statusok = save( obj, trial, options, paths )
            %SAVE Save experiment to disk
            %
            % Syntax:
            %
            % statusok = obj.save( trial ) save experiment trials
            % to disk as a *.mat file. 
            %
            % statusok = obj.save( __,"PARAM1", value1, ... ) specifies
            % optional parameter name/value pairs:
            %
            %   "Id" - index specific experiment items. Id is specified
            %   as integer scalar or array and corresponds to TrialID in
            %   the RowName.
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
                trial (1,:) {mustBeIntegerOrAll}
                options.Data (1,1) {mustBeMember( options.Data, ["original" "processed"] )} = "processed"
                options.Keys (1,:) string
                paths.WriteDirectory (1,1) string = fullfile( tempdir, "mlexperiment" )
                paths.FileName (1,1) string = ""
            end
            
            statusok = false;
            
            if isstring( trial )
                trial = obj.ind2sub( obj.describe.Item );
            end
            
            if ~isfolder( paths.WriteDirectory )
                mkdir( paths.WriteDirectory )
            end
            
            if ~isempty( obj.validTrialIndices )
                
                for iTrial = trial(:)'
                    
                    %Get info
                    full    = obj.describe( "full" );
                    trials  = obj.ind2sub( full.Item );
                    info    = full( trials == iTrial, : );
                    
                    args  = namedargs2cell( options );
                    
                    %Get data
                    result = obj.exportdata( iTrial, args{:});
                    
                    if ~isempty( result )
                        
                        if ~isfolder( paths.WriteDirectory )
                            mkdir( paths.WriteDirectory )
                        end
                        
                        if paths.FileName == ""
                            filename = "ExperimentML_" + "Trial" + iTrial + "_" ...
                                + string( datetime('now', 'format', 'MMMddyyyy') ) + ".mat";
                        else
                            filename = paths.FileName + "_" + "Trial" + iTrial + "_" ...
                                + string( datetime('now', 'format', 'MMMddyyyy') ) + ".mat";
                        end %if
                        
                        save( fullfile( paths.WriteDirectory, filename ), ...
                            "result", "info", ...
                            "-v7.3");
                        
                        statusok = true;
                        
                    end
                    
                end %for iTrial
            end
            
        end %function
        
        
        function [value, info] = select(obj, item, options, custom)
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
            % [value, ...] = obj.select("PARAM1", value1, ... ) specifies optional
            % parameter name/value pairs:
            %
            %   "Id"      - index to a specific experiment item. Id is specified
            %   as integer scalar and corresponds to TrialID in
            %   the RowName.
            %
            %   %Metadata - (default false). If true, returns a structure
            %   containing the model, model name , data configuration, model
            %   configuration, and a data table containing the result of
            %   the data preparation steps and fit predictions.
            %
            
            arguments
                obj experiment.Base
                item (1,1) double {mustBeInteger} = 1
                options.Metadata (1,1) logical = false
                custom.Data (1,1) {mustBeMember( custom.Data, ["" "original" "processed"] )} = "processed"
                custom.Keys (1,:) string = string.empty
            end

            args = namedargs2cell( options );
            [value, info] = select@experiment.Base( obj, item, args{:} );
            
            %If original data source is selected update the result
            if custom.Data ~= ""
                if ~isempty( obj.validTrialIndices )
                    
                    trial = obj.ind2sub( item );
                    value = obj.exportdata( trial, custom.Data, ...
                        "Keys", custom.Keys);
                end
            end
            
        end %function
        
        
        function plotmetric(obj, options)
            %PLOTMETRICS Plot silhouette
                        
            arguments
                obj
                options.Metric (1,1) string {mustBeMember(options.Metric,...
                    "silhouette")} = "silhouette"; 
                options.Axes = gobjects(0)
            end
            
            if isempty(options.Axes) && isgraphics(gca, 'axes')
                options.Axes = gca();
                
            % handles Chart case    
            elseif isempty(options.Axes) && ~isgraphics(gca, 'axes')
                figure("Color", "W");
                options.Axes = gca();
                
            else
                % Do nothing
            end
                     
            h = tiledlayout(options.Axes.Parent, "flow", "TileSpacing", "compact" );
            h.Parent.Color = "White";
            
            for iTrial = 1 : numel(obj.Trials)
            
                trial  = obj.Trials( iTrial );
                
                for iLabel = 1 : numel( trial.Label )
 
                    label = ("Label" + iLabel);

                    nexttile()
                    viz.silhouette( trial.Data, trial.features, label )

                    modeltype = extractBetween( trial.Label(iLabel).metadata.modelType,'(',')');
                    titlename = "trial "+ iTrial + ": " + modeltype;
                       
                    title( titlename )
                      
                end %for iLabel
            
            end %for iTrial
            
        end %function
        
        
        function build(obj)
           
             %Generate configurations
            if ~isempty( obj.DataConfiguration ) || ~isempty( obj.ModelConfiguration )
                obj.sample();
            end
            
            %Assign constants
            if ~isempty( obj.AdditionalConfiguration )
                obj.i_buildconstants();
            end
                        
            %Create experiment items
            trials = experiment.item.Unsupervised.empty(0,1);
            for i = 1:obj.nSamples
                trials(i) = experiment.item.Unsupervised();
                set(trials(i), 'Data', obj.Data);
            end
            set(trials, {'pipeline', 'model'}, {obj.DataFcn, obj.Model});
            set(trials, {'pipelinesettings'}, obj.i_pipelinesettings)
            set(trials, {'modelsettings'}, obj.i_modelsettings)
            
            %Assign the trials
            obj.Trials = trials;
            
        end %function
        
    end %public
    
    
    methods (Hidden)
        
        function resetmodels( obj )
            %RESETMODELS TODO DOC
            
            for i = 1:numel( obj.Trials )
                trial = obj.Trials(i);
                
                if ~isempty( trial.Label )
                    
                    trial.Label = [];
                    
                    tf = contains(trial.Data.Properties.VariableNames, "Prediction");
                    trial.Data( :, tf ) = [];

                end
                
            end %function
            
        end %function
        
    end %hidden methods
    
    
    methods ( Access = protected )
        function value = exportdata( obj, trial, data, options )
            
            arguments
                obj
                trial (1,1) double {mustBeInteger} = 1
                data (1,1) {mustBeMember( data,["original" "processed"] )} = "processed"
                options.Keys (1,:) string {mustBeValidKey( options.Keys, data )}% add custom validator to make sure Keys is not empty when Data is original
            end
  
            output = obj.view( "Id", trial );
            if data == "original"
                tF = contains(output.Properties.VariableNames, "Label");
                labelvars = string(output.Properties.VariableNames(tF));
                value = innerjoin( obj.Data, output(:, [options.Keys labelvars] ), "Keys", options.Keys );
            else
                value = output;
            end
            
        end %function
        
        
        function i_postprepareChecks( obj )
            
            for  i = 1:numel( obj.Trials )
                
                trial = obj.Trials(i);
                
                %Validate standard custom properties definition in preparation
                %pipeline
                tf_Feature  = isprop( trial.Data.Properties.CustomProperties, "Features" );
                tF_Options  = isprop( trial.Data.Properties.CustomProperties, "Options" );
                
                if ~tf_Feature
                    warning( ['List of features is undefined in data preparation pipeline. ' ...
                        'If undefined, all data variables are mapped/considered predictor variables except the last.'...
                        'Please see newPipelineTemplate for example of defining feature list.'])
                end %if tF_Feature
              
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
     
                datasummary = util.summarize( trial.Data(:, features));

                if datasummary.tF_Missing
                    warning( 'Training data contain NaN.' )
                end
                
                if datasummary.tF_Constant
                    warning( 'Training data contain constant variables.' )
                end
                
                if datasummary.tF_ConstantInClass == true
                    warning( 'Training data contain constant variables in one or more response classes.' )
                end
                
            end %for iTrial
            
        end %function
        
    end %protected
        
end %classdef


function mustBeIntegerOrAll( value )

    if isstring(value)
        if value ~= "all"
           error("Please specify numeric integers or ""all""") 
        end
    else
        mustBeInteger(value)
    end 
    
end %function

function mustBeValidKey( key, data )

if isstring(key) && isstring(data)
    if data == "original" && isempty(key)
        error("key:KeyNotEmpty","Please specify Key when appending labels to original data.")
    end
else
    error("Data and Key must be strings.")
end

end %function
